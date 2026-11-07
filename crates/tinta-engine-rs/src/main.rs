use std::io::Write;
use tinta_engine_rs::engine::Engine;
use tinta_engine_rs::manifest::Manifest;
use tinta_engine_rs::paths;
use tinta_engine_rs::protocol::{log, Output};
use tinta_engine_rs::threads;

fn main() {
    let threads = threads::detect();
    let models_root = paths::models_root();
    log(&format!(
        "starting version {}; {} CPUs, {} physical cores, asr {} threads, diarization {} threads; models at {}",
        env!("CARGO_PKG_VERSION"),
        threads.available,
        threads.physical,
        threads.asr,
        threads.diarization,
        models_root.display()
    ));

    let mut engine = Engine::new(
        Output::new(std::io::stdout()),
        Manifest::embedded(),
        models_root,
        threads,
    );
    if let Err(error) = engine.run(std::io::stdin().lock()) {
        log(&format!("cannot read stdin: {error}"));
        let _ = std::io::stderr().flush();
        std::process::exit(1);
    }
    log("stdin closed, exiting");
}
