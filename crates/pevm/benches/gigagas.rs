//! Benchmark mocked blocks that exceed 1 Gigagas.

// TODO: More fancy benchmarks & plots.

use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    sync::Arc,
    thread,
};

use alloy_primitives::{hex::ToHex, Address, B256, U160, U256};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pevm::{
    chain::PevmEthereum, execute_revm_sequential, Bytecodes, ChainState, EvmAccount, EvmCode,
    InMemoryStorage, Pevm,
};
use revm::primitives::{BlockEnv, Bytecode, SpecId, TransactTo, TxEnv};
use revme::cmd::statetest::models::{
    Env, SpecName, Test, TestSuite, TestUnit, TransactionParts, TxPartIndices,
};
use serde::Serialize;
use serde_json::json;

// Better project structure

/// common module
#[path = "../tests/common/mod.rs"]
pub mod common;

/// erc20 module
#[path = "../tests/erc20/mod.rs"]
pub mod erc20;

/// uniswap module
#[path = "../tests/uniswap/mod.rs"]
pub mod uniswap;

///  large gas value
const GIGA_GAS: u64 = 1_000_000_000;

#[cfg(feature = "global-alloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Runs a benchmark for executing a set of transactions on a given blockchain state.
pub fn bench(c: &mut Criterion, name: &str, storage: InMemoryStorage, txs: Vec<TxEnv>) {
    let concurrency_level = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let chain = PevmEthereum::mainnet();
    let spec_id = SpecId::SHANGHAI;
    let block_env = BlockEnv::default();
    let mut pevm = Pevm::default();
    let mut group = c.benchmark_group(name);

    dump_test_suite(&block_env, &storage, &txs, name.into());
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            execute_revm_sequential(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
            )
        })
    });
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            pevm.execute_revm_parallel(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
                black_box(concurrency_level),
            )
        })
    });

    group.bench_function("Parallel Single", |b| {
        let txs = vec![txs[0].clone()];
        b.iter(|| {
            pevm.execute_revm_parallel(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
                black_box(concurrency_level),
            )
        })
    });
    group.finish();
}

/// Benchmarks the execution time of raw token transfers.
pub fn bench_raw_transfers(c: &mut Criterion) {
    let block_size = (GIGA_GAS as f64 / common::RAW_TRANSFER_GAS_LIMIT as f64).ceil() as usize;
    // Skip the built-in precompiled contracts addresses.
    const START_ADDRESS: usize = 1000;
    const MINER_ADDRESS: usize = 0;
    let storage = InMemoryStorage::new(
        std::iter::once(MINER_ADDRESS)
            .chain(START_ADDRESS..START_ADDRESS + block_size)
            .map(common::mock_account)
            .collect(),
        Default::default(),
        Default::default(),
    );
    bench(
        c,
        "Independent Raw Transfers",
        storage,
        (0..block_size)
            .map(|i| {
                let address = Address::from(U160::from(START_ADDRESS + i));
                TxEnv {
                    caller: address,
                    transact_to: TransactTo::Call(address),
                    value: U256::from(1),
                    gas_limit: common::RAW_TRANSFER_GAS_LIMIT,
                    gas_price: U256::from(1),
                    ..TxEnv::default()
                }
            })
            .collect::<Vec<_>>(),
    );
}

/// Benchmarks the execution time of ERC-20 token transfers.
pub fn bench_erc20(c: &mut Criterion) {
    let block_size = (GIGA_GAS as f64 / erc20::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let (mut state, bytecodes, txs) = erc20::generate_cluster(block_size, 1, 1);
    state.insert(Address::ZERO, EvmAccount::default()); // Beneficiary

    bench(
        c,
        "Independent ERC20",
        InMemoryStorage::new(state, Arc::new(bytecodes), Default::default()),
        txs,
    );
}

/// Benchmarks the execution time of Uniswap V3 swap transactions.
pub fn bench_uniswap(c: &mut Criterion) {
    let block_size = (GIGA_GAS as f64 / uniswap::ESTIMATED_GAS_USED as f64).ceil() as usize;
    let mut final_state = ChainState::from_iter([(Address::ZERO, EvmAccount::default())]); // Beneficiary
    let mut final_bytecodes = Bytecodes::default();
    let mut final_txs = Vec::<TxEnv>::new();
    for _ in 0..block_size {
        let (state, bytecodes, txs) = uniswap::generate_cluster(1, 1);
        final_state.extend(state);
        final_bytecodes.extend(bytecodes);
        final_txs.extend(txs);
    }
    bench(
        c,
        "Independent Uniswap",
        InMemoryStorage::new(final_state, Arc::new(final_bytecodes), Default::default()),
        final_txs,
    );
}

#[derive(Debug, Clone, Default, Serialize)]
struct PreAccount {
    balance: String,
    nonce: String,
    code: String,
    storage: HashMap<String, String>,
}

fn dump_test_suite(
    block_env: &BlockEnv,
    storage: &InMemoryStorage,
    txs: &[TxEnv],
    filename: String,
) {
    use std::fs::File;
    use std::io::Write;

    let json_dir = std::path::PathBuf::from("json").join(&filename);

    std::fs::create_dir_all(&json_dir).expect("Failed to create directory");

    let mut first = true;

    for tx in txs {
        let tx = tx.clone();
        let output_json = json_dir.join(format!(
            "{}0x{:032x}.json",
            {
                if first {
                    "head"
                } else {
                    ""
                }
            },
            tx.caller
        ));

        let mut file = File::create(output_json).unwrap();

        let mut pre: HashMap<String, PreAccount> = HashMap::default();

        if first {
            storage.accounts.iter().for_each(|(address, account)| {
                let mut storage_map = HashMap::default();
                account.storage.iter().for_each(|(key, value)| {
                    if !value.eq(&U256::default()) {
                        storage_map.insert(format!("0x{:064x}", key), format!("0x{:064x}", value));
                    }
                });

                let code: String = if let Some(ref codehash) = account.code_hash {
                    storage
                        .bytecodes
                        .get(codehash)
                        .map(|code| match code {
                            EvmCode::Legacy(code) => format!("{:0x}", code.bytecode),
                            EvmCode::Eof(code) => format!("{:0x}", code), // 0x prefix added by hex formatter
                            EvmCode::Eip7702(_) => panic!("Not supported"),
                        })
                        .unwrap_or_default()
                } else {
                    "".into()
                };

                pre.insert(
                    format!("0x{:032x}", address),
                    PreAccount {
                        balance: format!("0x{:064x}", account.balance),
                        nonce: format!("0x{:032x}", account.nonce),
                        code,
                        storage: storage_map,
                    },
                );
            });
        }

        let block_gas_limit = block_env.gas_limit.saturating_to::<u64>();

        let pre = if first { json!(pre) } else { json!({}) };

        let json = json!({
        "pevm_auto_test": json!(
            {
                "env": json!({
                    "currentCoinbase": block_env.coinbase,
                    "currentDifficulty": block_env.difficulty,
                    "currentGasLimit": block_gas_limit,
                    "currentNumber": block_env.number,
                    "currentTimestamp": block_env.timestamp,
                    "currentBaseFee": block_env.basefee,
                    "currentRandom": block_env.prevrandao,
                }),
                "pre": pre,
                "post": json!({
                    "Shanghai": json!([{
                        "indexes": {
                            "data": 0,
                            "gas": 0,
                            "value": 0
                        },
                        "hash": "0xffaaffee2df50c04d0247829676adce5898a419e0c00de9a76141765e22fdbc9",
                        "logs": "0xffaaffee2df50c04d0247829676adce5898a419e0c00de9a76141765e22fdbc9",
                    }])}),
                "transaction": json!(
                    {"data": [tx.data],
                     "gas_limit": [tx.gas_limit],
                     "gas_price": tx.gas_price,
                     "nonce": tx.nonce.unwrap_or_default(),
                     "sender": tx.caller,
                     "to": tx.transact_to.to().unwrap(),
                     "value": [format!("0x{:064x}", tx.value)],
                     "secretKey": "0xffaaffeed060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8",
                     "gasLimit": [format!("0x{:032x}", tx.gas_limit)],
                    }
                ),

            })});
        file.write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes())
            .unwrap();
        first = false;
    }
}

/// Runs a series of benchmarks to evaluate the performance of different transaction types.
pub fn benchmark_gigagas(c: &mut Criterion) {
    // bench_raw_transfers(c);
    // bench_erc20(c);
    bench_uniswap(c);
}

// HACK: we can't document public items inside of the macro
#[allow(missing_docs)]
mod benches {
    use super::*;
    criterion_group!(benches, benchmark_gigagas);
}

criterion_main!(benches::benches);
