use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::State;
use uuid::Uuid;

use super::UiState;

const CHAT_HISTORY_VERSION: u32 = 1;
const NEW_CHAT_TITLE: &str = "New chat";
const MAX_THREAD_TITLE_LENGTH: usize = 48;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ChatHistory {
    version: u32,
    active_thread_id: String,
    threads: Vec<ChatThread>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatThread {
    id: String,
    title: String,
    messages: Vec<ChatMessage>,
    created_at: u64,
    updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatMessage {
    id: String,
    role: ChatRole,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ChatRole {
    User,
    Assistant,
}

pub(crate) struct ChatHistoryStore {
    path: PathBuf,
    document: ChatHistory,
}

enum DocumentParseError {
    Corrupt(String),
    UnsupportedVersion(u64),
}

impl ChatHistory {
    fn empty(now: u64) -> Self {
        let thread = ChatThread::new(now);
        Self {
            version: CHAT_HISTORY_VERSION,
            active_thread_id: thread.id.clone(),
            threads: vec![thread],
        }
    }
}

impl ChatThread {
    fn new(now: u64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: NEW_CHAT_TITLE.to_owned(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl ChatHistoryStore {
    pub(crate) fn open(path: PathBuf) -> Result<Self, String> {
        match fs::read(&path) {
            Ok(bytes) => match parse_document(&bytes) {
                Ok(document) => Ok(Self { path, document }),
                Err(DocumentParseError::UnsupportedVersion(version)) => Err(format!(
                    "unsupported chat history version {version}; expected {CHAT_HISTORY_VERSION}"
                )),
                Err(DocumentParseError::Corrupt(reason)) => {
                    let quarantined = quarantine_corrupt_file(&path).map_err(|error| {
                        format!(
                            "failed to quarantine malformed chat history at {}: {error}",
                            path.display()
                        )
                    })?;
                    let store = Self {
                        path,
                        document: ChatHistory::empty(now_unix_ms()),
                    };
                    store.persist(&store.document).map_err(|error| {
                        format!(
                            "recovered malformed chat history ({reason}) to {}, but failed to save a replacement after moving the original to {}: {error}",
                            store.path.display(),
                            quarantined.display()
                        )
                    })?;
                    Ok(store)
                }
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let store = Self {
                    path,
                    document: ChatHistory::empty(now_unix_ms()),
                };
                store.persist(&store.document)?;
                Ok(store)
            }
            Err(error) => Err(format!(
                "failed to read chat history at {}: {error}",
                path.display()
            )),
        }
    }

    fn snapshot(&self) -> ChatHistory {
        self.document.clone()
    }

    fn create_thread(&mut self) -> Result<ChatHistory, String> {
        let thread = ChatThread::new(now_unix_ms());
        let mut next = self.document.clone();
        next.active_thread_id = thread.id.clone();
        next.threads.insert(0, thread);
        self.commit(next)
    }

    fn select_thread(&mut self, thread_id: &str) -> Result<ChatHistory, String> {
        if !self
            .document
            .threads
            .iter()
            .any(|thread| thread.id == thread_id)
        {
            return Err(format!("chat thread not found: {thread_id}"));
        }

        let mut next = self.document.clone();
        next.active_thread_id = thread_id.to_owned();
        self.commit(next)
    }

    fn append_message(
        &mut self,
        thread_id: &str,
        role: ChatRole,
        text: String,
    ) -> Result<ChatHistory, String> {
        let Some(thread_index) = self
            .document
            .threads
            .iter()
            .position(|thread| thread.id == thread_id)
        else {
            return Err(format!("chat thread not found: {thread_id}"));
        };

        let mut next = self.document.clone();
        let mut thread = next.threads.remove(thread_index);
        let should_create_title = role == ChatRole::User
            && !thread
                .messages
                .iter()
                .any(|message| message.role == ChatRole::User);
        if should_create_title {
            thread.title = thread_title_from_prompt(&text);
        }
        thread.messages.push(ChatMessage {
            id: Uuid::new_v4().to_string(),
            role,
            text,
        });
        thread.updated_at = now_unix_ms();
        next.threads.insert(0, thread);
        self.commit(next)
    }

    fn delete_thread(&mut self, thread_id: &str) -> Result<ChatHistory, String> {
        let Some(thread_index) = self
            .document
            .threads
            .iter()
            .position(|thread| thread.id == thread_id)
        else {
            return Err(format!("chat thread not found: {thread_id}"));
        };

        let mut next = self.document.clone();
        let deleted_active_thread = next.active_thread_id == thread_id;
        next.threads.remove(thread_index);

        if next.threads.is_empty() {
            next = ChatHistory::empty(now_unix_ms());
        } else if deleted_active_thread {
            let next_active_index = thread_index.min(next.threads.len() - 1);
            next.active_thread_id = next.threads[next_active_index].id.clone();
        }

        self.commit(next)
    }

    fn commit(&mut self, next: ChatHistory) -> Result<ChatHistory, String> {
        self.persist(&next)?;
        self.document = next;
        Ok(self.document.clone())
    }

    fn persist(&self, document: &ChatHistory) -> Result<(), String> {
        persist_document(&self.path, document).map_err(|error| {
            format!(
                "failed to persist chat history at {}: {error}",
                self.path.display()
            )
        })
    }
}

#[tauri::command]
pub(crate) fn get_chat_history(state: State<'_, UiState>) -> Result<ChatHistory, String> {
    with_store(&state, |store| Ok(store.snapshot()))
}

#[tauri::command]
pub(crate) fn create_chat_thread(state: State<'_, UiState>) -> Result<ChatHistory, String> {
    with_store(&state, ChatHistoryStore::create_thread)
}

#[tauri::command]
pub(crate) fn select_chat_thread(
    state: State<'_, UiState>,
    thread_id: String,
) -> Result<ChatHistory, String> {
    with_store(&state, |store| store.select_thread(&thread_id))
}

#[tauri::command]
pub(crate) fn append_chat_message(
    state: State<'_, UiState>,
    thread_id: String,
    role: ChatRole,
    text: String,
) -> Result<ChatHistory, String> {
    with_store(&state, |store| store.append_message(&thread_id, role, text))
}

#[tauri::command]
pub(crate) fn delete_chat_thread(
    state: State<'_, UiState>,
    thread_id: String,
) -> Result<ChatHistory, String> {
    with_store(&state, |store| store.delete_thread(&thread_id))
}

fn with_store<T>(
    state: &State<'_, UiState>,
    operation: impl FnOnce(&mut ChatHistoryStore) -> Result<T, String>,
) -> Result<T, String> {
    let mut history = state
        .chat_history
        .lock()
        .map_err(|_| "failed to lock chat history".to_owned())?;
    let store = history
        .as_mut()
        .ok_or_else(|| "chat history is not initialized".to_owned())?;
    operation(store)
}

fn parse_document(bytes: &[u8]) -> Result<ChatHistory, DocumentParseError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| DocumentParseError::Corrupt(error.to_string()))?;

    match value.get("version").and_then(Value::as_u64) {
        Some(version) if version == u64::from(CHAT_HISTORY_VERSION) => {}
        Some(version) => return Err(DocumentParseError::UnsupportedVersion(version)),
        None => {
            return Err(DocumentParseError::Corrupt(
                "missing or invalid version".to_owned(),
            ));
        }
    }

    let document: ChatHistory = serde_json::from_value(value)
        .map_err(|error| DocumentParseError::Corrupt(error.to_string()))?;
    validate_document(&document).map_err(DocumentParseError::Corrupt)?;
    Ok(document)
}

fn validate_document(document: &ChatHistory) -> Result<(), String> {
    if document.version != CHAT_HISTORY_VERSION {
        return Err("version changed during deserialization".to_owned());
    }
    if document.threads.is_empty() {
        return Err("history has no threads".to_owned());
    }
    if !document
        .threads
        .iter()
        .any(|thread| thread.id == document.active_thread_id)
    {
        return Err("active thread does not exist".to_owned());
    }

    let mut thread_ids = HashSet::new();
    let mut message_ids = HashSet::new();
    for thread in &document.threads {
        if thread.id.trim().is_empty() || !thread_ids.insert(&thread.id) {
            return Err("history contains an empty or duplicate thread id".to_owned());
        }
        for message in &thread.messages {
            if message.id.trim().is_empty() || !message_ids.insert(&message.id) {
                return Err("history contains an empty or duplicate message id".to_owned());
            }
        }
    }

    Ok(())
}

fn thread_title_from_prompt(prompt: &str) -> String {
    let normalized = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return NEW_CHAT_TITLE.to_owned();
    }

    let characters = normalized.chars().collect::<Vec<_>>();
    if characters.len() <= MAX_THREAD_TITLE_LENGTH {
        return normalized;
    }

    let prefix = characters[..MAX_THREAD_TITLE_LENGTH - 1]
        .iter()
        .collect::<String>();
    format!("{prefix}…")
}

fn persist_document(path: &Path, document: &ChatHistory) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("chat-history-v1.json");
    let temporary_path = parent.join(format!(".{file_name}.tmp-{}", Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let write_result = (|| {
        let mut file = options.open(&temporary_path)?;
        serde_json::to_writer_pretty(&mut file, document)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        replace_with_rename(&temporary_path, path)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result
}

#[cfg(not(windows))]
fn replace_with_rename(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_with_rename(source: &Path, destination: &Path) -> io::Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(_) if destination.exists() => {
            fs::remove_file(destination)?;
            fs::rename(source, destination)
        }
        Err(error) => Err(error),
    }
}

fn quarantine_corrupt_file(path: &Path) -> io::Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("chat-history-v1.json");
    let quarantine_path = parent.join(format!(
        "{file_name}.corrupt-{}-{}",
        now_unix_ms(),
        Uuid::new_v4()
    ));
    fs::rename(path, &quarantine_path)?;
    Ok(quarantine_path)
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("infernet-chat-history-{label}-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).expect("create chat history test directory");
            Self(path)
        }

        fn history_path(&self) -> PathBuf {
            self.0.join("chat-history-v1.json")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn missing_file_creates_one_persisted_thread() {
        let directory = TestDirectory::new("missing");
        let store = ChatHistoryStore::open(directory.history_path()).expect("open history");

        assert_eq!(store.document.version, 1);
        assert_eq!(store.document.threads.len(), 1);
        assert_eq!(
            store.document.active_thread_id,
            store.document.threads[0].id
        );
        assert!(directory.history_path().is_file());
        let serialized: Value = serde_json::from_slice(
            &fs::read(directory.history_path()).expect("read persisted history"),
        )
        .expect("parse persisted history");
        assert!(serialized.get("activeThreadId").is_some());
        assert!(serialized["threads"][0].get("createdAt").is_some());
        assert!(serialized["threads"][0].get("updatedAt").is_some());
    }

    #[test]
    fn create_makes_a_new_active_thread_at_the_front() {
        let directory = TestDirectory::new("create");
        let mut store = ChatHistoryStore::open(directory.history_path()).expect("open history");
        let original_id = store.document.active_thread_id.clone();

        let document = store.create_thread().expect("create thread");

        assert_eq!(document.threads.len(), 2);
        assert_eq!(document.active_thread_id, document.threads[0].id);
        assert_ne!(document.active_thread_id, original_id);
        assert_eq!(document.threads[0].title, NEW_CHAT_TITLE);
    }

    #[test]
    fn append_titles_first_user_prompt_without_splitting_unicode() {
        let directory = TestDirectory::new("append");
        let mut store = ChatHistoryStore::open(directory.history_path()).expect("open history");
        let thread_id = store.document.active_thread_id.clone();
        store
            .append_message(&thread_id, ChatRole::Assistant, "Hello".to_owned())
            .expect("append assistant response");
        let prompt = format!("  {}   done  ", "界".repeat(60));

        let document = store
            .append_message(&thread_id, ChatRole::User, prompt.clone())
            .expect("append user prompt");
        let thread = &document.threads[0];

        assert_eq!(thread.id, thread_id);
        assert_eq!(thread.messages.len(), 2);
        assert_eq!(thread.messages[1].text, prompt);
        assert_eq!(thread.title.chars().count(), MAX_THREAD_TITLE_LENGTH);
        assert!(thread.title.ends_with('…'));
    }

    #[test]
    fn delete_chooses_the_next_thread_and_replaces_the_last_one() {
        let directory = TestDirectory::new("delete");
        let mut store = ChatHistoryStore::open(directory.history_path()).expect("open history");
        let original_id = store.document.active_thread_id.clone();
        let second = store.create_thread().expect("create second thread");
        let second_id = second.active_thread_id.clone();
        let third = store.create_thread().expect("create third thread");
        let third_id = third.active_thread_id.clone();
        store
            .select_thread(&second_id)
            .expect("select middle thread");

        let after_middle_delete = store
            .delete_thread(&second_id)
            .expect("delete middle thread");
        assert_eq!(after_middle_delete.active_thread_id, original_id);
        assert_eq!(
            after_middle_delete
                .threads
                .iter()
                .map(|thread| thread.id.as_str())
                .collect::<Vec<_>>(),
            vec![third_id.as_str(), original_id.as_str()]
        );

        store.delete_thread(&third_id).expect("delete third thread");
        let replacement = store
            .delete_thread(&original_id)
            .expect("delete final thread");
        assert_eq!(replacement.threads.len(), 1);
        assert_eq!(replacement.active_thread_id, replacement.threads[0].id);
        assert_ne!(replacement.active_thread_id, original_id);
        assert!(replacement.threads[0].messages.is_empty());
    }

    #[test]
    fn mutations_survive_reopen() {
        let directory = TestDirectory::new("reopen");
        let path = directory.history_path();
        let expected = {
            let mut store = ChatHistoryStore::open(path.clone()).expect("open history");
            let thread_id = store.document.active_thread_id.clone();
            store
                .append_message(&thread_id, ChatRole::User, "Persistent prompt".to_owned())
                .expect("append persisted message")
        };

        let reopened = ChatHistoryStore::open(path).expect("reopen history");
        assert_eq!(reopened.document, expected);
    }

    #[test]
    fn malformed_json_is_quarantined_and_recovered() {
        let directory = TestDirectory::new("corrupt");
        let path = directory.history_path();
        let malformed = b"{ definitely not json";
        fs::write(&path, malformed).expect("write malformed history");

        let store = ChatHistoryStore::open(path.clone()).expect("recover history");

        assert_eq!(store.document.threads.len(), 1);
        assert!(path.is_file());
        let quarantine = fs::read_dir(&directory.0)
            .expect("list test directory")
            .filter_map(Result::ok)
            .find(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("chat-history-v1.json.corrupt-")
            })
            .expect("find quarantined history");
        assert_eq!(
            fs::read(quarantine.path()).expect("read quarantine"),
            malformed
        );
    }

    #[test]
    fn unsupported_version_is_not_overwritten_or_quarantined() {
        let directory = TestDirectory::new("version");
        let path = directory.history_path();
        let future = br#"{"version":2,"activeThreadId":"future","threads":[]}"#;
        fs::write(&path, future).expect("write future history");

        let error = ChatHistoryStore::open(path.clone())
            .err()
            .expect("reject unsupported version");

        assert!(error.contains("unsupported chat history version 2"));
        assert_eq!(fs::read(&path).expect("reread future history"), future);
        assert_eq!(
            fs::read_dir(&directory.0)
                .expect("list test directory")
                .count(),
            1
        );
    }
}
