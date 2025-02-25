//! Benchmark mainnet blocks with needed state loaded in memory.

// TODO: More fancy benchmarks & plots.
use std::{fs::File, io::BufReader, num::NonZeroUsize, sync::Arc, thread};

use alloy_primitives::{keccak256, Address, TxKind, U256};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use pevm::{
    chain::PevmEthereum, BlockHashes, BuildSuffixHasher, Bytecodes, EvmAccount, InMemoryStorage,
    Pevm,
};
use revm::primitives::{
    Account, AccountInfo, BlockEnv, Bytecode, EvmStorage, EvmStorageSlot, SpecId, TxEnv,
};
use revme::cmd::statetest::models::TestSuite;

// Better project structure

/// common module
#[path = "../tests/common/mod.rs"]
pub mod common;

// [rpmalloc] is generally better but can crash on ARM.
#[cfg(feature = "global-alloc")]
#[cfg(target_arch = "aarch64")]
#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;
#[cfg(feature = "global-alloc")]
#[cfg(not(target_arch = "aarch64"))]
#[global_allocator]
static GLOBAL: rpmalloc::RpMalloc = rpmalloc::RpMalloc;

/// Benchmark for ERC transactions
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut pevm = Pevm::default();
    let chain = PevmEthereum::mainnet();
    let concurrency_level = thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .min(
            NonZeroUsize::new(
                #[cfg(target_arch = "aarch64")]
                12,
                #[cfg(not(target_arch = "aarch64"))]
                8,
            )
            .unwrap(),
        );

    let data_dir = std::path::PathBuf::from("../../data");
    let erc20_transfer_basic_usdt_json = data_dir
        .join("benchmark")
        .join("erc20_transfer_basic_usdt.json");

    let suite: TestSuite = serde_json::from_reader(BufReader::new(
        File::open(erc20_transfer_basic_usdt_json).unwrap(),
    ))
    .unwrap();

    let test = suite.0;
    let test = test.first_key_value().unwrap().1;

    let mut bytecodes: Bytecodes = Bytecodes::default();
    let mut accounts: HashMap<Address, EvmAccount, BuildSuffixHasher> = HashMap::default();
    test.pre.iter().for_each(|(address, account)| {
        let code_hash = keccak256(&account.code);
        let code = match account.code.len() > 0 {
            true => Some(Bytecode::new_raw(account.code.clone())),
            false => None,
        };

        let storage: EvmStorage = account
            .storage
            .iter()
            .map(|(k, v)| {
                (
                    *k,
                    EvmStorageSlot {
                        original_value: *v,
                        present_value: *v,
                        is_cold: true,
                    },
                )
            })
            .collect();

        let account = Account {
            info: AccountInfo {
                nonce: account.nonce,
                balance: account.balance,
                code,
                code_hash,
            },
            storage,
            ..Default::default()
        };
        accounts.insert(*address, account.into());
    });

    // iter over accounts and get the bytecodes
    accounts.iter().for_each(|(_address, account)| {
        if let Some(ref bytecode) = account.code.clone() {
            bytecodes.insert(account.code_hash.unwrap(), bytecode.clone());
        }
    });

    let bytecodes = Arc::new(bytecodes);

    let block_hashes: BlockHashes = BlockHashes::default();
    let block_hashes = Arc::new(block_hashes);

    let mut block_env = BlockEnv::default();
    let spec_id = SpecId::SHANGHAI;

    block_env.basefee = test.env.current_base_fee.unwrap();
    block_env.difficulty = test.env.current_difficulty;
    block_env.timestamp = test.env.current_timestamp;
    block_env.coinbase = test.env.current_coinbase;
    block_env.gas_limit = test.env.current_gas_limit;
    block_env.number = test.env.current_number;

    let mut tx = TxEnv {
        caller: test.transaction.sender.unwrap(),
        ..Default::default()
    };

    tx.gas_price = test.transaction.gas_price.unwrap();
    tx.gas_priority_fee = test.transaction.max_priority_fee_per_gas;
    tx.blob_hashes = test.transaction.blob_versioned_hashes.clone();
    tx.max_fee_per_blob_gas = test.transaction.max_fee_per_blob_gas;
    tx.data = test.transaction.data[0].clone();
    tx.gas_limit = test.transaction.gas_limit[0].try_into().unwrap();
    let value = &test.transaction.value[0];
    tx.value = if let Some(stripped) = value.strip_prefix("0x") {
        U256::from_str_radix(stripped, 16).unwrap()
    } else {
        U256::from_str_radix(value, 16).unwrap()
    };
    tx.nonce = u64::try_from(test.transaction.nonce).ok();
    let to = match test.transaction.to {
        Some(add) => TxKind::Call(add),
        None => TxKind::Create,
    };
    tx.transact_to = to;

    let txs = vec![tx];
    let storage = InMemoryStorage::new(accounts, Arc::clone(&bytecodes), Arc::clone(&block_hashes));

    c.bench_function("parallel_execute_erc_function", |b| {
        b.iter(|| {
            pevm.execute_revm_parallel(
                black_box(&chain),
                black_box(&storage),
                black_box(spec_id),
                black_box(block_env.clone()),
                black_box(txs.clone()),
                black_box(concurrency_level),
            )
            .unwrap();
        })
    });
}

// HACK: we can't document public items inside of the macro
#[allow(missing_docs)]
mod benches {
    use super::*;
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(benches::benches);
