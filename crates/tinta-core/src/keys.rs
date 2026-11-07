use anyhow::{anyhow, Context, Result};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;

/// The service of the master key in the system key store.
const SERVICE: &str = "com.usetinta.tinta";
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
        Self::load_or_create_with(&SystemStore)
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

    /// Uses a store for the master key. Only a clear "no entry" result creates a key: any other
    /// failure would replace a key that is still there, and make the library unreadable.
    #[cfg(any(target_os = "linux", target_os = "windows", test))]
    fn load_or_create_with(store: &dyn KeyStore) -> Result<Self> {
        let value = match store.get()? {
            Some(value) => value,
            None => {
                let mut key = [0u8; 32];
                rand::thread_rng().fill_bytes(&mut key);
                let value = hex::encode(key);
                store.set(&value)?;
                value
            }
        };
        Self::derive(&hex::decode(&value).context("invalid master key in the system key store")?)
    }

    /// Test keys that never touch the Keychain.
    pub fn for_tests() -> Self {
        Self::derive(&[7u8; 32]).expect("derive")
    }
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
        match entry.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(anyhow!(
                "cannot read the library key from the system key store: {error}"
            )),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

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

    #[test]
    fn first_run_creates_and_stores_a_key() {
        let store = MockStore::empty();
        let keys = Keys::load_or_create_with(&store).expect("load or create");
        let stored = store.value().expect("a new key is stored");
        assert_eq!(store.writes(), 1);
        assert_eq!(stored.len(), 64, "the master key is 32 bytes of hex");
        assert!(stored.chars().all(|c| c.is_ascii_hexdigit()));
        let master = hex::decode(&stored).expect("hex");
        assert_eq!(
            keys.database,
            Keys::derive(&master).expect("derive").database
        );
    }

    #[test]
    fn second_run_reuses_the_stored_key() {
        let stored = "ab".repeat(32);
        let store = MockStore::with_value(&stored);
        let keys = Keys::load_or_create_with(&store).expect("load");
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some(stored.as_str()));
        let expected = Keys::derive(&hex::decode(&stored).expect("hex")).expect("derive");
        assert_eq!(keys.audio, expected.audio);
    }

    #[test]
    fn locked_store_errors_without_writing() {
        let store = MockStore::with_read_error("the key store is locked");
        let error = Keys::load_or_create_with(&store)
            .err()
            .expect("a locked store must not create a key");
        assert!(error.to_string().contains("locked"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn unavailable_store_errors_without_writing() {
        let store = MockStore::with_read_error("no Secret Service is running");
        let error = Keys::load_or_create_with(&store)
            .err()
            .expect("an unavailable store must not create a key");
        assert!(error.to_string().contains("Secret Service"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }

    #[test]
    fn malformed_stored_value_errors_without_writing() {
        let store = MockStore::with_value("not hex");
        let error = Keys::load_or_create_with(&store)
            .err()
            .expect("a malformed key must not create a key");
        assert!(error.to_string().contains("invalid master key"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value().as_deref(), Some("not hex"));
    }

    #[test]
    fn store_write_failure_errors() {
        let store = MockStore::with_write_error("access denied");
        let error = Keys::load_or_create_with(&store)
            .err()
            .expect("a failed write must fail");
        assert!(error.to_string().contains("access denied"));
        assert_eq!(store.writes(), 0);
        assert_eq!(store.value(), None);
    }
}
