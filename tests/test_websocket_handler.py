# pytest tests/test_websocket_handler.py -v

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from websocket_handler import agent_process_message


@pytest.fixture
def mock_agent():
    """Фикстура для мока агента."""
    return AsyncMock()


class TestAgentProcessMessage:
    """Тесты для функции agent_process_message."""

    @pytest.mark.asyncio
    async def test_basic_message_processing(self, mock_agent):
        """Тест базовой обработки сообщения с сохранением SystemMessage."""

        user_message = "Как дела?"
        expected_answer = "Всё отлично, спасибо!"

        # История содержит системное сообщение — проверяем, что оно сохранится первым
        sample_chat_history = [SystemMessage(content="system prompt")]

        mock_response = {"messages": [
            SystemMessage(content="system"),
            HumanMessage(content="test"),
            AIMessage(content=expected_answer)
        ]}
        mock_agent.ainvoke.return_value = mock_response

        answer, updated_history = await agent_process_message(
            agent=mock_agent,
            user_message=user_message,
            chat_history=sample_chat_history.copy()
        )

        # Assert
        assert answer == expected_answer
        assert len(updated_history) == 3  # System + Human + AI
        assert isinstance(updated_history[0], SystemMessage)
        assert updated_history[0].content == "system prompt"
        assert isinstance(updated_history[-2], HumanMessage)
        assert updated_history[-2].content == user_message
        assert isinstance(updated_history[-1], AIMessage)
        assert updated_history[-1].content == expected_answer

        # Проверяем, что в вызов агенту SystemMessage передан первым
        call_args = mock_agent.ainvoke.call_args[0][0]
        assert isinstance(call_args["messages"][0], SystemMessage)
        assert call_args["messages"][0].content == "system prompt"

    @pytest.mark.asyncio
    async def test_history_limit_enforced(self, mock_agent):
        """Тест ограничения размера истории чата."""

        with patch("websocket_handler.MAX_CHAT_HISTORY", 2):

            chat_history = []

            # Создаём историю из 6 пар (12 сообщений)
            for i in range(6):
                chat_history.append(HumanMessage(content=f"Вопрос {i}"))
                chat_history.append(AIMessage(content=f"Ответ {i}"))

            mock_response = {"messages": [AIMessage(content="Ответ")]}
            mock_agent.ainvoke.return_value = mock_response

            # Act
            answer, updated_history = await agent_process_message(
                agent=mock_agent,
                user_message="Новый вопрос",
                chat_history=chat_history
            )

            # После добавления новых сообщений история = 14, обрезается до 4
            assert len(updated_history) == 4
            assert updated_history[0].content == "Вопрос 5"
            assert updated_history[1].content == "Ответ 5"
            assert updated_history[-2].content == "Новый вопрос"
            assert updated_history[-1].content == "Ответ"

    @pytest.mark.asyncio
    async def test_history_limit_not_triggered(self, mock_agent):
        """Тест, что ограничение длины истории не срабатывает при малом количестве сообщений."""

        with patch("websocket_handler.MAX_CHAT_HISTORY", 10):
            chat_history = [HumanMessage(content="Привет"), AIMessage(content="Hi")]

            mock_response = {"messages": [AIMessage(content="Ответ")]}
            mock_agent.ainvoke.return_value = mock_response

            answer, updated_history = await agent_process_message(
                agent=mock_agent,
                user_message="Вопрос",
                chat_history=chat_history
            )

            assert len(updated_history) == 4  # 2 + 2 новых, меньше лимита 20

    @pytest.mark.asyncio
    async def test_empty_chat_history(self, mock_agent):
        """Тест обработки с пустой историей чата."""

        user_message = "Первое сообщение"
        chat_history = []

        mock_response = {"messages": [AIMessage(content="Первый ответ")]}
        mock_agent.ainvoke.return_value = mock_response

        answer, updated_history = await agent_process_message(
            agent=mock_agent,
            user_message=user_message,
            chat_history=chat_history
        )

        assert answer == "Первый ответ"
        assert len(updated_history) == 2
        assert updated_history[0].content == user_message
        assert updated_history[1].content == "Первый ответ"

    @pytest.mark.asyncio
    async def test_agent_exception_propagated(self, mock_agent):
        """Тест проброса исключений от агента."""

        mock_agent.ainvoke.side_effect = Exception("LLM Error")

        with pytest.raises(Exception, match="LLM Error"):
            await agent_process_message(
                agent=mock_agent,
                user_message="test",
                chat_history=[]
            )


class TestEdgeCases:
    """Тесты edge cases."""

    @pytest.mark.asyncio
    async def test_history_at_exact_limit(self, mock_agent):
        """Тест когда исходная история точно на границе лимита."""

        with patch("websocket_handler.MAX_CHAT_HISTORY", 2):

            chat_history = [
                HumanMessage(content="1"), AIMessage(content="1"),
                HumanMessage(content="2"), AIMessage(content="2"),
            ]
            mock_response = {"messages": [AIMessage(content="3")]}
            mock_agent.ainvoke.return_value = mock_response

            answer, history = await agent_process_message(
                agent=mock_agent,
                user_message="3",
                chat_history=chat_history
            )

            # Было 4 (лимит), добавили 2 = 6, обрезали до 4
            assert len(history) == 4
            assert history[0].content == "2"
            assert history[1].content == "2"
            assert history[2].content == "3"
            assert history[3].content == "3"
