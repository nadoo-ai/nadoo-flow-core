"""
Tests for memory_db module (Database-backed chat history with Redis caching)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nadoo_flow.memory_db import (
    DatabaseAdapter,
    DatabaseChatHistory,
    create_database_history_manager,
)
from nadoo_flow.prompts import Message


@pytest.fixture
def mock_db_adapter():
    """Mock DatabaseAdapter for testing"""
    adapter = AsyncMock(spec=DatabaseAdapter)
    adapter.save_message = AsyncMock()
    adapter.get_messages = AsyncMock(return_value=[])
    adapter.clear_messages = AsyncMock()
    return adapter


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    redis = MagicMock()
    redis.lrange = MagicMock(return_value=[])
    redis.rpush = MagicMock()
    redis.delete = MagicMock()
    redis.expire = MagicMock()
    return redis


@pytest.mark.asyncio
async def test_database_chat_history_init(mock_db_adapter):
    """Test DatabaseChatHistory initialization"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    assert history.session_id == "test-session"
    assert history.workspace_id == "workspace-1"
    assert history.db_adapter == mock_db_adapter
    assert history.redis is None


@pytest.mark.asyncio
async def test_add_message_db_only(mock_db_adapter):
    """Test adding message to DB (no Redis)"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    message = Message(role="user", content="Hello")
    await history.add_message(message)

    # Verify DB save was called
    mock_db_adapter.save_message.assert_called_once_with(
        session_id="test-session",
        message=message,
        workspace_id="workspace-1"
    )


@pytest.mark.asyncio
async def test_add_message_with_redis(mock_db_adapter, mock_redis):
    """Test adding message to both DB and Redis"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    message = Message(role="user", content="Hello")
    await history.add_message(message)

    # Verify DB save
    mock_db_adapter.save_message.assert_called_once()

    # Verify Redis cache
    assert mock_redis.rpush.called
    assert mock_redis.expire.called


@pytest.mark.asyncio
async def test_get_messages_from_db(mock_db_adapter):
    """Test getting messages from DB (no Redis)"""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]
    mock_db_adapter.get_messages.return_value = messages

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    result = await history.get_messages()

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    mock_db_adapter.get_messages.assert_called_once_with(
        session_id="test-session",
        workspace_id="workspace-1"
    )


@pytest.mark.asyncio
async def test_get_messages_redis_cache_hit(mock_db_adapter, mock_redis):
    """Test getting messages from Redis cache (cache hit)"""
    import json

    # Mock Redis cache hit
    cached_messages = [
        json.dumps({"role": "user", "content": "Hello", "metadata": {}}),
        json.dumps({"role": "assistant", "content": "Hi!", "metadata": {}})
    ]
    mock_redis.lrange.return_value = cached_messages

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    result = await history.get_messages()

    # Should return from cache, not call DB
    assert len(result) == 2
    assert result[0].role == "user"
    mock_db_adapter.get_messages.assert_not_called()


@pytest.mark.asyncio
async def test_get_messages_redis_cache_miss(mock_db_adapter, mock_redis):
    """Test getting messages from DB when Redis cache misses"""
    # Mock Redis cache miss
    mock_redis.lrange.return_value = []

    # Mock DB has messages
    db_messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ]
    mock_db_adapter.get_messages.return_value = db_messages

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    result = await history.get_messages()

    # Should call DB on cache miss
    assert len(result) == 2
    mock_db_adapter.get_messages.assert_called_once()

    # Should cache results to Redis
    assert mock_redis.rpush.called


@pytest.mark.asyncio
async def test_redis_key_generation():
    """Test Redis key generation for workspace isolation"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=AsyncMock(),
        redis_client=MagicMock(),
        workspace_id="workspace-1"
    )

    key = history._make_redis_key()

    # Should include workspace_id for isolation
    assert "workspace-1" in key
    assert "test-session" in key


@pytest.mark.asyncio
async def test_create_database_history_manager(mock_db_adapter, mock_redis):
    """Test create_database_history_manager factory function"""
    manager = create_database_history_manager(
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        default_workspace_id="workspace-1"
    )

    # Should return SessionHistoryManager
    assert hasattr(manager, "get_history")

    # Test creating history for a session
    history = await manager.get_history("session-1")
    assert history.session_id == "session-1"


@pytest.mark.asyncio
async def test_ttl_configuration(mock_db_adapter, mock_redis):
    """Test Redis TTL configuration"""
    ttl = 7200  # 2 hours

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1",
        redis_ttl=ttl
    )

    message = Message(role="user", content="Hello")
    await history.add_message(message)

    # Verify TTL was set
    mock_redis.expire.assert_called()
    call_args = mock_redis.expire.call_args
    assert call_args[0][1] == ttl  # Check TTL value


@pytest.mark.asyncio
async def test_workspace_isolation(mock_db_adapter):
    """Test that different workspaces are isolated"""
    history1 = DatabaseChatHistory(
        session_id="session-1",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    history2 = DatabaseChatHistory(
        session_id="session-1",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-2"
    )

    message = Message(role="user", content="Test")
    await history1.add_message(message)

    # Should call with different workspace_id
    mock_db_adapter.save_message.assert_called_with(
        session_id="session-1",
        message=message,
        workspace_id="workspace-1"
    )


@pytest.mark.asyncio
async def test_error_handling_db_failure(mock_db_adapter):
    """Test error handling when DB fails"""
    mock_db_adapter.save_message.side_effect = Exception("DB Error")

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    message = Message(role="user", content="Hello")

    # Should raise exception on DB failure
    with pytest.raises(Exception) as exc_info:
        await history.add_message(message)

    assert "DB Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_redis_failure_fallback(mock_db_adapter, mock_redis):
    """Test fallback to DB when Redis fails"""
    # Mock Redis failure
    mock_redis.lrange.side_effect = Exception("Redis Error")

    # Mock DB has messages
    db_messages = [Message(role="user", content="Hello")]
    mock_db_adapter.get_messages.return_value = db_messages

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    # Should fallback to DB on Redis failure
    result = await history.get_messages()

    assert len(result) == 1
    mock_db_adapter.get_messages.assert_called_once()


@pytest.mark.asyncio
async def test_message_metadata_preservation(mock_db_adapter):
    """Test that message metadata is preserved through DB"""
    metadata = {
        "tool_calls": [{"name": "search", "args": {"query": "test"}}],
        "retrieved_chunks": ["chunk-1", "chunk-2"],
        "total_tokens": 150
    }

    message = Message(role="assistant", content="Result", metadata=metadata)

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        workspace_id="workspace-1"
    )

    await history.add_message(message)

    # Verify metadata was passed to DB
    call_args = mock_db_adapter.save_message.call_args
    saved_message = call_args.kwargs["message"]
    assert saved_message.metadata == metadata


@pytest.mark.asyncio
async def test_redis_key_without_workspace():
    """Test Redis key generation without workspace_id"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=AsyncMock(),
        redis_client=MagicMock(),
        workspace_id=None  # No workspace
    )

    key = history._make_redis_key()

    # Should NOT include workspace_id
    assert "test-session" in key
    assert key == "chat_history:test-session"


@pytest.mark.asyncio
async def test_get_messages_cache_and_store(mock_db_adapter, mock_redis):
    """Test that DB messages are cached to Redis"""
    # Mock Redis cache miss
    mock_redis.lrange.return_value = []

    # Mock DB has messages
    db_messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ]
    mock_db_adapter.get_messages.return_value = db_messages

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    result = await history.get_messages()

    # Should return from DB
    assert len(result) == 2

    # Should cache to Redis (delete + rpush for each message + expire)
    assert mock_redis.delete.called
    assert mock_redis.rpush.call_count == 2
    assert mock_redis.expire.called


@pytest.mark.asyncio
async def test_add_messages_batch(mock_db_adapter, mock_redis):
    """Test adding multiple messages in batch"""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!"),
        Message(role="user", content="How are you?")
    ]

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    await history.add_messages(messages)

    # Verify DB batch save was called
    mock_db_adapter.save_messages.assert_called_once_with(
        session_id="test-session",
        messages=messages,
        workspace_id="workspace-1"
    )

    # Verify Redis caching
    assert mock_redis.delete.called
    assert mock_redis.rpush.call_count == 3


@pytest.mark.asyncio
async def test_clear_history(mock_db_adapter, mock_redis):
    """Test clearing chat history"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    await history.clear()

    # Verify DB clear was called
    mock_db_adapter.clear_messages.assert_called_once_with(
        session_id="test-session",
        workspace_id="workspace-1"
    )

    # Verify Redis clear
    assert mock_redis.delete.called


@pytest.mark.asyncio
async def test_clear_history_redis_failure(mock_db_adapter, mock_redis):
    """Test that clear continues even if Redis fails"""
    # Mock Redis failure
    mock_redis.delete.side_effect = Exception("Redis Error")

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    # Should not raise exception
    await history.clear()

    # DB should still be cleared
    mock_db_adapter.clear_messages.assert_called_once()


@pytest.mark.asyncio
async def test_add_message_redis_failure_non_critical(mock_db_adapter, mock_redis):
    """Test that add_message continues if Redis caching fails"""
    # Mock Redis failure
    mock_redis.rpush.side_effect = Exception("Redis Error")

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    message = Message(role="user", content="Hello")

    # Should not raise exception (Redis failure is non-critical)
    await history.add_message(message)

    # DB should still be saved
    mock_db_adapter.save_message.assert_called_once()


@pytest.mark.asyncio
async def test_add_messages_batch_redis_failure(mock_db_adapter, mock_redis):
    """Test that add_messages continues if Redis caching fails"""
    # Mock Redis failure during caching
    mock_redis.delete.side_effect = Exception("Redis Error")

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ]

    # Should not raise exception
    await history.add_messages(messages)

    # DB should still be saved
    mock_db_adapter.save_messages.assert_called_once()


@pytest.mark.asyncio
async def test_get_messages_empty_cache_empty_db(mock_db_adapter, mock_redis):
    """Test get_messages when both cache and DB are empty"""
    # Mock Redis cache miss
    mock_redis.lrange.return_value = []

    # Mock DB has no messages
    mock_db_adapter.get_messages.return_value = []

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    result = await history.get_messages()

    # Should return empty list
    assert len(result) == 0

    # Should not cache empty results
    assert not mock_redis.rpush.called


@pytest.mark.asyncio
async def test_cache_messages_to_redis_failure(mock_db_adapter, mock_redis):
    """Test that get_messages handles Redis caching failure gracefully"""
    # Mock Redis cache miss
    mock_redis.lrange.return_value = []

    # Mock DB has messages
    db_messages = [Message(role="user", content="Hello")]
    mock_db_adapter.get_messages.return_value = db_messages

    # Mock Redis caching failure (_cache_messages_to_redis will raise)
    mock_redis.delete.side_effect = Exception("Redis caching failed")

    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=mock_redis,
        workspace_id="workspace-1"
    )

    # Should not raise exception (caching failure is logged but not fatal)
    result = await history.get_messages()

    # Should still return messages from DB
    assert len(result) == 1
    assert result[0].content == "Hello"


@pytest.mark.asyncio
async def test_cache_messages_to_redis_no_redis(mock_db_adapter):
    """Test _cache_messages_to_redis when redis_client is None"""
    history = DatabaseChatHistory(
        session_id="test-session",
        db_adapter=mock_db_adapter,
        redis_client=None,  # No Redis
        workspace_id="workspace-1"
    )

    messages = [Message(role="user", content="Hello")]

    # Should not raise exception when Redis is None
    # (_cache_messages_to_redis should return early)
    await history._cache_messages_to_redis(messages)
