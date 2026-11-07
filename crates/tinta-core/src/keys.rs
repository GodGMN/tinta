use anyhow::{anyhow, Result};
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows", test))]
use anyhow::Context;
use hkdf::Hkdf;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows", test))]
use rand::RngCore;
use sha2::Sha256;

/// The service of the master key in the system key store.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const SERVICE: &str = "com.usetinta.tinta";
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
const ACCOUNT: &str = "library-master-key";

/// The service of the master key in versions before 0.4.0.
#[cfg(target_os = "macos")]
const OLD_SERVICE: &str = "app.tinta";
/// errSecItemNotFound
#[cfg(target_os = "macos")]
const ITEM_NOT_FOUND: i32 = -25300;

/// Keys derived from the master key in the system key store.
#[derive(Clone)]
pub struct Keys {
    pub database: [u8; 32],
    pub audio: [u8; 32],
}

impl Keys {
    pub fn derive(master: &[u8]) -> Result<Self> {
        let hk = Hkdf::<Sha256>::new(Some(b"tinta"), master);
        let mut database = [0u8; 32];
        let mut audio = [0u8; 32];
        hk.expand(b"database", &mut database).map_err(|_| anyhow!("hkdf"))?;
        hk.expand(b"audio", &mut audio).map_err(|_| anyhow!("hkdf"))?;
        Ok(Self { database, audio })
    }

    pub fn audio_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(self.audio)
    }

    /// Loads the master key from the Keychain, or creates it on first use.
    /// The item is a generic password that iCloud Keychain does not synchronize.
    #[cfg(target_os = "macos")]
    pub fn load_or_create() -> Result<Self> {
        use security_framework::passwords::set_generic_password;
        // Debug builds can use a fixed key, so that automated tests do not wait for Keychain prompts.
        #[cfg(debug_assertions)]
        if let Ok(value) = std::env::var("TINTA_DEV_MASTER_KEY") {
            return Self::derive(&hex::decode(value).context("TINTA_DEV_MASTER_KEY must be hex")?);
        }
        let master = match read_item(SERVICE)? {
            Some(value) => value,
            None => match read_item(OLD_SERVICE)? {
                // Moves the key of an older version to the current service, once. The app creates
                // the new item, so macOS does not ask for access to it again.
                Some(value) => {
                    set_generic_password(SERVICE, ACCOUNT, &value)
                        .context("cannot store the master key in the Keychain")?;
                    let _ = security_framework::passwords::delete_generic_password(OLD_SERVICE, ACCOUNT);
                    value
                }
                None => {
                    let mut key = [0u8; 32];
                    rand::thread_rng().fill_bytes(&mut key);
                    let value = hex::encode(key).into_bytes();
                    set_generic_password(SERVICE, ACCOUNT, &value)
                        .context("cannot store the master key in the Keychain")?;
                    value
                }
            },
        };
        Self::derive(&hex::decode(master).context("invalid master key in Keychain")?)
    }

    /// Loads the master key from the system key store, or creates it on first use.
    /// Linux uses the Secret Service (GNOME Keyring or KWallet); Windows uses the Credential
    /// Manager. The item has the same service and account as the macOS Keychain item.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn load_or_create() -> Result<Self> {
        // Debug builds can use a fixed key, like on macOS, so tests and the self-test never
        // touch (or, with a locked keyring, overwrite) the user's real key.
        #[cfg(debug_assertions)]
        if let Ok(value) = std::env::var("TINTA_DEV_MASTER_KEY") {
            return Self::derive(&hex::decode(value).context("TINTA_DEV_MASTER_KEY must be hex")?);
        }
        // There is one key per user, so the lock and the "existing library" check also cover
        // the fixed per-user folder, whatever TINTA_DATA_DIR or the configured library says.
        let user_dir = dirs::data_dir()
            .ok_or_else(|| anyhow!("cannot find the user data folder"))?
            .join("Tinta");
        let library_dirs = [crate::paths::data_dir(), user_dir.clone()];
        Self::load_or_create_in(&user_dir, &library_dirs, &SystemStore)
    }

    /// A non-macOS build for a platform without a secure store. It refuses to create a key,
    /// because a key that cannot be stored would not survive a restart.
    #[cfg(all(
        not(target_os = "macos"),
        not(target_os = "linux"),
        not(target_os = "windows")
    ))]
    pub fn load_or_create() -> Result<Self> {
        Err(anyhow!(
            "the library key store is not supported on this platform"
        ))
    }

    /// Uses a store for the master key. Two instances must not both create a key, so the whole
    /// read/create/write runs under an exclusive lock of the library folder. Only a clear "no
    /// entry" result creates a key, and only when the library is still empty: any other failure,
    /// or an existing library without a stored key, would make the library unreadable.
    #[cfg(any(target_os = "linux", target_os = "windows", test))]
    fn load_or_create_in(
        lock_dir: &std::path::Path,
        library_dirs: &[std::path::PathBuf],
        store: &dyn KeyStore,
    ) -> Result<Self> {
        let _lock = lock_library(lock_dir)?;
        let value = match store.get()? {
            Some(value) => value,
            None => {
                if library_dirs.iter().any(|dir| has_existing_data(dir)) {
                    return Err(anyhow!(
                        "The key for your existing library was not found. Unlock your keyring \
                         (GNOME Keyring, KWallet, KeePassXC) and open Tinta again."
                    ));
                }
                let mut key = [0u8; 32];
                rand::thread_rng().fill_bytes(&mut key);
                let written = hex::encode(key);
                store.set(&written)?;
                // A racing writer, or a store that accepted the write but dropped it, must win:
                // the database must use exactly the key that the store now holds.
                store.get()?.ok_or_else(|| {
                    anyhow!("the system key store did not keep the new library key")
                })?
            }
        };
        Self::derive(&decode_master_key(&value)?)
    }

    /// Test keys that never touch the Keychain.
    pub fn for_tests() -> Self {
        Self::derive(&[7u8; 32]).expect("derive")
    }
}

/// Decodes the stored hex key and requires exactly 32 bytes. A short or empty value must never
/// be used (or replaced): the store may hold it because another app wrote the item.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn decode_master_key(value: &str) -> Result<[u8; 32]> {
    let master = hex::decode(value).context("invalid master key in the system key store")?;
    master.try_into().map_err(|_| {
        anyhow!("invalid master key in the system key store (expected 32 bytes)")
    })
}

/// True when the folder may already hold a library: any `library.db*` file (including WAL and
/// SHM), anything under `meetings/`, or a `config.json` that points at a moved library (which
/// may sit on an unmounted drive). Any error other than "not found" counts as existing data,
/// because guessing "empty" could create a second key.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn has_existing_data(dir: &std::path::Path) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => return error.kind() != std::io::ErrorKind::NotFound,
    };
    for entry in entries {
        let Ok(entry) = entry else { return true };
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("library.db") {
            return true;
        }
        if name == "config.json" && configured_library(&entry.path()) {
            return true;
        }
        if name == "meetings" {
            match std::fs::read_dir(entry.path()) {
                Ok(mut meetings) => {
                    if meetings.next().is_some() {
                        return true;
                    }
                }
                Err(error) => return error.kind() != std::io::ErrorKind::NotFound,
            }
        }
    }
    false
}

/// True when a `config.json` names a library folder, or cannot be read.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn configured_library(config: &std::path::Path) -> bool {
    match std::fs::read_to_string(config) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text)
            .map(|v| !v["library_dir"].is_null())
            .unwrap_or(true),
        Err(_) => true,
    }
}

/// Holds an exclusive lock on the library folder until the returned file is dropped.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
fn lock_library(dir: &std::path::Path) -> Result<std::fs::File> {
    std::fs::create_dir_all(dir).context("cannot create the library folder")?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("key.lock"))
        .context("cannot open the library key lock file")?;
    file.lock().context("cannot lock the library folder")?;
    Ok(file)
}

/// Reads the master key item of a service. Only a missing item gives `None`. Any other error,
/// for example a denied Keychain prompt, must stop the app: a new key would make the library unreadable.
#[cfg(target_os = "macos")]
fn read_item(service: &str) -> Result<Option<Vec<u8>>> {
    match security_framework::passwords::get_generic_password(service, ACCOUNT) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.code() == ITEM_NOT_FOUND => Ok(None),
        Err(error) => Err(anyhow!("cannot read the library key from the Keychain: {error}")),
    }
}

/// A secure store that holds the hex-encoded master key.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
trait KeyStore {
    /// Reads the stored value. Returns `None` only when the store clearly has no entry. A locked
    /// store, a missing service, a denied prompt, and every other failure are errors: creating a
    /// key then would make the existing library unreadable.
    fn get(&self) -> Result<Option<String>>;

    /// Stores the value, replacing the previous one.
    fn set(&self, value: &str) -> Result<()>;
}

/// The platform key store: Secret Service on Linux, Credential Manager on Windows.
#[cfg(any(target_os = "linux", target_os = "windows"))]
struct SystemStore;

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl KeyStore for SystemStore {
    fn get(&self) -> Result<Option<String>> {
        let entry =
            keyring::Entry::new(SERVICE, ACCOUNT).context("cannot open the system key store")?;
        read_password(&entry)
    }

    fn set(&self, value: &str) -> Result<()> {
        keyring::Entry::new(SERVICE, ACCOUNT)
            .context("cannot open the system key store")?
            .set_password(value)
            .map_err(|error| {
                anyhow!("cannot store the library key in the system key store: {error}")
            })
    }
}

/// Maps the errors of the real key store. Only `NoEntry` means the item is absent; a locked or
/// unreachable store, an ambiguous match, and a malformed value are all errors.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn read_password(entry: &keyring::Entry) -> Result<Option<String>> {
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(anyhow!(
            "cannot read the library key from the system key store: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn load_in(dir: &std::path::Path, store: &dyn KeyStore) -> Result<Keys> {
        Keys::load_or_create_in(dir, &[dir.to_path_buf()], store)
    }

    /// A store where another instance wins the write: reads after a write return its key.
    struct RacingStore {
        winner: String,
        written: Mutex<bool>,
    }

    impl KeyStore for RacingStore {
        fn get(&self) -> Result<Option<String>> {
            Ok(self.written.lock().unwrap().then(|| self.winner.clone()))
        }
        fn set(&self, _value: &str) -> Result<()> {
            *self.written.lock().unwrap() = true;
            Ok(())
        }
    }

    #[test]
    fn uses_the_stored_key_when_another_instance_wins() {
        let dir = tempfile::tempdir().unwrap();
        let store = RacingStore { winner: hex::encode([9u8; 32]), written: Mutex::new(false) };
        let keys = load_in(dir.path(), &store).unwrap();
        assert_eq!(keys.database, Keys::derive(&[9u8; 32]).unwrap().database);
    }

    #[test]
    fn wal_files_meetings_and_a_moved_library_count_as_existing_data() {
        for (name, contents) in [
            ("library.db-wal", "x"),
            ("library.db-shm", "x"),
            ("config.json", r#"{"library_dir":"/mnt/usb/Tinta"}"#),
            ("config.json", "not json"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(name), contents).unwrap();
            assert!(has_existing_data(dir.path()), "{name}: {contents}");
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("meetings/m1")).unwrap();
        assert!(has_existing_data(dir.path()), "meeting folder without audio");
    }

    #[test]
    fn any_library_folder_with_data_blocks_creation() {
        let lock = tempfile::tempdir().unwrap();
        let empty = tempfile::tempdir().unwrap();
        let real = tempfile::tempdir().unwrap();
        std::fs::write(real.path().join("library.db"), "x").unwrap();
        let store = MockStore::empty();
        let dirs = [empty.path().to_path_buf(), real.path().to_path_buf()];
        assert!(Keys::load_or_create_in(lock.path(), &dirs, &store).is_err());
        assert_eq!(store.writes(), 0);
    }

    struct MockStore {
        value: Mutex<Option<String>>,
        read_error: Option<&'static str>,
        write_error: Option<&'static str>,
        writes: Mutex<usize>,
    }

    impl MockStore {
        fn empty() -> Self {
            Self {
                value: Mutex::new(None),
                read_error: None,
                write_error: None,
                writes: Mutex::new(0),
            }
        }

        fn with_value(value: &str) -> Self {
            Self {
                value: Mutex::new(Some(value.to_string())),
                ..Self::empty()
            }
        }

        fn with_read_error(message: &'static str) -> Self {
            Self {
                read_error: Some(message),
                ..Self::empty()
            }
        }

        fn with_write_error(message: &'static str) -> Self {
            Self {
                write_error: Some(message),
                ..Self::empty()
            }
        }

        fn value(&self) -> Option<String> {
            self.value.lock().unwrap().clone()
        }

        fn writes(&self) -> usize {
            *self.writes.lock().unwrap()
        }
    }

    impl KeyStore for MockStore {
        fn get(&self) -> Result<Option<String>> {
            if let Some(message) = self.read_error {
                return Err(anyhow!("{message}"));
            }
            Ok(self.value())
        }

        fn set(&self, value: &str) -> Result<()> {
            if let Some(message) = self.write_error {
                return Err(anyhow!("{message}"));
            }
            *self.writes.lock().unwrap() += 1;
            *self.value.lock().unwrap() = Some(value.to_string());
            Ok(())
        }
    }

    fn library() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp library")
    }

    fn stored_master(value: &str) -> [u8; 32] {
        hex::decode(value).expect("hex").try_into().expect("32 bytes")
    }

    #[test]
    fn first_run_creates_and_stores_a_key() {
        let dir = library();
        let store = MockStore::empty();
        let keys = load_in(dir.path(), &store).expect("load or create");
        let stored = store.value().expect("a new key is stored");
        assert_eq!(store.writes(), 1);
        assert_eq!(stored.len(), 64, "the master key is 32 bytes of hex");
        assert!(stored.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(keys.database, Keys::derive(&stored_master(&stored)).expect("derive").database);
    }

    #[test]
    fn second_run_reuses_the_stored_key() {
        let dir = library();
        let stored = "ab".repeat(32);
        let store = MockStore::with_value(&stored);
        let keys = load_in(dir.path(), &store).expect("load");
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some(stored.as_str()));
        let expected = Keys::derive(&stored_master(&stored)).expect("derive");
        assert_eq!(keys.audio, expected.audio);
    }

    #[test]
    fn existing_library_with_a_stored_key_loads() {
        let dir = library();
        std::fs::write(dir.path().join("library.db"), b"encrypted library").expect("library");
        let stored = "cd".repeat(32);
        let store = MockStore::with_value(&stored);
        let keys = load_in(dir.path(), &store).expect("load");
        assert_eq!(store.writes(), 0);
        assert_eq!(keys.database, Keys::derive(&stored_master(&stored)).expect("derive").database);
    }

    #[test]
    fn existing_library_without_a_key_errors() {
        let dir = library();
        std::fs::write(dir.path().join("library.db"), b"encrypted library").expect("library");
        let store = MockStore::empty();
        let error = load_in(dir.path(), &store)
            .err()
            .expect("an existing library must not get a new key");
        assert!(error.to_string().contains("key for your existing library"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn existing_audio_without_a_key_errors() {
        let dir = library();
        let audio = dir.path().join("meetings").join("m1").join("audio");
        std::fs::create_dir_all(&audio).expect("audio dir");
        std::fs::write(audio.join("0.wav"), b"encrypted audio").expect("audio");
        let store = MockStore::empty();
        let error = load_in(dir.path(), &store)
            .err()
            .expect("existing recordings must not get a new key");
        assert!(error.to_string().contains("key for your existing library"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn malformed_stored_value_errors_without_writing() {
        let dir = library();
        let store = MockStore::with_value("not hex");
        let error = load_in(dir.path(), &store)
            .err()
            .expect("a malformed key must not create a key");
        assert!(error.to_string().contains("invalid master key"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some("not hex"));
    }

    #[test]
    fn empty_stored_value_errors_without_writing() {
        let dir = library();
        let store = MockStore::with_value("");
        let error = load_in(dir.path(), &store)
            .err()
            .expect("an empty key must not be used");
        assert!(error.to_string().contains("expected 32 bytes"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some(""));
    }

    #[test]
    fn short_stored_value_errors_without_writing() {
        let dir = library();
        let short = "ab".repeat(31);
        let store = MockStore::with_value(&short);
        let error = load_in(dir.path(), &store)
            .err()
            .expect("a 31-byte key must not be used");
        assert!(error.to_string().contains("expected 32 bytes"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some(short.as_str()));
    }

    #[test]
    fn locked_store_errors_without_writing() {
        let dir = library();
        let store = MockStore::with_read_error("the key store is locked");
        let error = load_in(dir.path(), &store)
            .err()
            .expect("a locked store must not create a key");
        assert!(error.to_string().contains("locked"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn unavailable_store_errors_without_writing() {
        let dir = library();
        let store = MockStore::with_read_error("no Secret Service is running");
        let error = load_in(dir.path(), &store)
            .err()
            .expect("an unavailable store must not create a key");
        assert!(error.to_string().contains("Secret Service"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn store_write_failure_errors() {
        let dir = library();
        let store = MockStore::with_write_error("access denied");
        let error = load_in(dir.path(), &store)
            .err()
            .expect("a failed write must fail");
        assert!(error.to_string().contains("access denied"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn dropped_write_errors_instead_of_using_the_lost_key() {
        struct DroppingStore(MockStore);

        impl KeyStore for DroppingStore {
            fn get(&self) -> Result<Option<String>> {
                self.0.get()
            }

            fn set(&self, _value: &str) -> Result<()> {
                *self.0.writes.lock().unwrap() += 1;
                Ok(())
            }
        }

        let dir = library();
        let store = DroppingStore(MockStore::empty());
        let error = load_in(dir.path(), &store)
            .err()
            .expect("a store that drops the write must fail");
        assert!(error.to_string().contains("did not keep"));
        assert_eq!(store.0.writes(), 1);
        assert_eq!(store.0.value(), None);
    }

    #[test]
    fn two_threads_create_one_key() {
        use std::sync::Arc;

        let dir = library();
        let store = Arc::new(MockStore::empty());
        let path = dir.path().to_path_buf();
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let store = Arc::clone(&store);
                let path = path.clone();
                std::thread::spawn(move || {
                    load_in(&path, store.as_ref()).expect("load or create")
                })
            })
            .collect();
        let keys: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
        assert_eq!(store.writes(), 1, "only one instance creates the key");
        assert_eq!(keys[0].database, keys[1].database);
        assert_eq!(keys[0].audio, keys[1].audio);
        let stored = store.value().expect("stored key");
        assert_eq!(keys[0].database, Keys::derive(&stored_master(&stored)).expect("derive").database);
    }

    #[test]
    fn the_lock_waits_for_the_holder() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir = library();
        let held = lock_library(dir.path()).expect("lock");
        let path = dir.path().to_path_buf();
        let (sender, receiver) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _lock = lock_library(&path).expect("lock");
            sender.send(()).expect("send");
        });
        assert!(
            receiver.recv_timeout(Duration::from_millis(100)).is_err(),
            "a second instance must wait for the lock"
        );
        drop(held);
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the lock is released");
        thread.join().expect("thread");
    }

    /// Exercises the mapping of the real `keyring` errors, using its mock credential builder.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn with_mock_error(error: Option<keyring::Error>) -> Result<Option<String>> {
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap();
        keyring::set_default_credential_builder(keyring::mock::default_credential_builder());
        let entry = keyring::Entry::new(SERVICE, ACCOUNT).expect("mock entry");
        if let Some(error) = error {
            let mock: &keyring::mock::MockCredential =
                entry.get_credential().downcast_ref().expect("mock credential");
            mock.set_error(error);
        }
        read_password(&entry)
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn only_no_entry_means_the_key_is_absent() {
        assert_eq!(
            with_mock_error(None).expect("no entry").as_deref(),
            None,
            "NoEntry maps to absent"
        );

        let errors = [
            keyring::Error::PlatformFailure("no session bus".into()),
            keyring::Error::NoStorageAccess("the collection is locked".into()),
            keyring::Error::BadEncoding(vec![0xff, 0xfe]),
            keyring::Error::Ambiguous(vec![
                Box::new(keyring::mock::MockCredential::default())
            ]),
            keyring::Error::TooLong("service".to_string(), 10),
            keyring::Error::Invalid("service".to_string(), "bad".to_string()),
        ];
        for error in errors {
            assert!(
                with_mock_error(Some(error)).is_err(),
                "only NoEntry may mean the key is absent"
            );
        }
    }

    /// Creates a key through a real Secret Service. Ignored by default: it needs a
    /// running `org.freedesktop.secrets` provider and empty data folders. The
    /// `integration-smoke` job runs it inside a `dbus-run-session` with
    /// `gnome-keyring-daemon --unlock`.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "needs a running Secret Service; set TINTA_SECRET_SERVICE_TEST=1"]
    fn creates_a_library_key_through_the_secret_service() {
        if std::env::var("TINTA_SECRET_SERVICE_TEST").as_deref() != Ok("1") {
            return;
        }
        let entry = keyring::Entry::new(SERVICE, ACCOUNT).expect("entry");
        // Start from no item, so the test proves creation rather than reuse.
        let _ = entry.delete_credential();

        let created = Keys::load_or_create().expect("create the key in the Secret Service");
        let stored = entry
            .get_password()
            .expect("read the key back from the Secret Service");
        assert_eq!(stored.len(), 64, "the master key is 32 bytes as hex");
        assert_eq!(
            created.database,
            Keys::derive(&decode_master_key(&stored).expect("stored key"))
                .expect("derive")
                .database
        );

        // A second load reads the stored item instead of writing another key.
        let again = Keys::load_or_create().expect("read the existing key");
        assert_eq!(created.database, again.database);

        let _ = entry.delete_credential();
    }
}
