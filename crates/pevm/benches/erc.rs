//! Benchmark mainnet blocks with needed state loaded in memory.

// TODO: More fancy benchmarks & plots.
use std::{collections::BTreeMap, fs::File, io::BufReader, num::NonZeroUsize, sync::Arc, thread};

use alloy_primitives::{keccak256, Address, Bytes};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use pevm::{
    chain::PevmEthereum, BlockHashes, BuildSuffixHasher, Bytecodes, EvmAccount, InMemoryStorage,
    Pevm,
};
use revm::primitives::{BlockEnv, Env, SpecId};
use revme::cmd::statetest::models::{SpecName, Test, TransactionParts};
use serde::Deserialize;

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

/// A single test unit.
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestUnit {
    /// Test info is optional
    #[serde(default, rename = "_info")]
    pub info: Option<serde_json::Value>,

    pub env: Env,
    pub pre: HashMap<Address, EvmAccount, BuildSuffixHasher>,
    pub post: BTreeMap<SpecName, Vec<Test>>,
    pub transaction: TransactionParts,
    #[serde(default)]
    pub out: Option<Bytes>,
}

/// The top level test suite.
#[derive(Debug, PartialEq, Eq, Deserialize)]
struct TestSuite(pub BTreeMap<String, TestUnit>);

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

    println!("test_unit: {:?}", test);
    let accounts: HashMap<Address, EvmAccount, BuildSuffixHasher> = test.pre;

    let mut bytecodes: Bytecodes = Bytecodes::default();

    // iter over accounts and get the bytecodes
    for (address, account) in accounts.iter() {
        if let Some(bytecode) = account.code.clone() {
            let code_hash = account
                .code_hash
                .unwrap_or_else(|| keccak256(&bytecode.clone().into()));
            bytecodes.insert(code_hash, bytecode);
        }
    }

    let bytecodes = Arc::new(bytecodes);

    let block_hashes: BlockHashes = BlockHashes::default();
    let block_hashes = Arc::new(block_hashes);

    let block_env = BlockEnv::default();
    let spec_id = SpecId::SHANGHAI;

    let txs = todo!();

    c.bench_function("parallel_execute_erc_function", |b| {
        b.iter(|| {
            let storage =
                InMemoryStorage::new(accounts, Arc::clone(&bytecodes), Arc::clone(&block_hashes));
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
}

// HACK: we can't document public items inside of the macro
#[allow(missing_docs)]
mod benches {
    use super::*;
    criterion_group!(benches, criterion_benchmark);
}

criterion_main!(benches::benches);
