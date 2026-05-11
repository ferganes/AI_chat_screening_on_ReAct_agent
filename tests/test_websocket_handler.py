# pytest tests/test_websocket_handler.py -v

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from websocket_handler import agent_process_message


class TestAgentProcessMessage:
    """Тесты для функции agent_process_message."""

    @pytest.mark.asyncio
    async def test_basic_message_processing(self, mock_agent, sample_chat_history, system_prompt):
        """Тест базовой обработки сообщения."""
        # Arrange
        user_message = "Как дела?"
        expected_answer = "Всё отлично, спасибо!"

        mock_response = {"messages": [
            SystemMessage(content="system"),
            HumanMessage(content="test"),
            AIMessage(content=expected_answer)
        ]}
        mock_agent.ainvoke.return_value = mock_response

        # Act
        with patch("websocket_handler.SYSTEM_PROMPT", system_prompt):
            answer, updated_history = await agent_process_message(
                agent=mock_agent,
                user_message=user_message,
                chat_history=sample_chat_history.copy()
            )

        # Assert
        assert answer == expected_answer
        assert len(updated_history) == 4
        assert isinstance(updated_history[-2], HumanMessage)
        assert updated_history[-2].content == user_message
        assert isinstance(updated_history[-1], AIMessage)
        assert updated_history[-1].content == expected_answer

        # Проверяем, что в вызов агенту SystemMessage передан первым
        call_args = mock_agent.ainvoke.call_args[0][0]
        assert isinstance(call_args["messages"][0], SystemMessage)
        assert call_args["messages"][0].content == system_prompt

    @pytest.mark.asyncio
    async def test_history_limit_enforced(self, mock_agent):
        """Тест ограничения размера истории чата."""
        # Arrange
        with patch("websocket_handler.MAX_CHAT_HISTORY", 2):

            chat_history = []

            # Создаём историю больше лимита
            for i in range(6):  # 6 сообщений = 3 пары, лимит 2 пары
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

            # Assert
            # после добавления новых сообщений история = 8, обрезается до 4
            # Но обрезка происходит ПОСЛЕ добавления, поэтому остаются последние 4
            assert len(updated_history) == 4  # MAX_CHAT_HISTORY * 2 = 4
            assert updated_history[0].content == "Вопрос 5"
            assert updated_history[-2].content == "Новый вопрос"

    @pytest.mark.asyncio
    async def test_history_limit_not_triggered(self, mock_agent):
        """Тест, что ограничение не срабатывает при малой истории."""
        # Arrange
        with patch("websocket_handler.MAX_CHAT_HISTORY", 10):
            chat_history = [HumanMessage(content="Привет"), AIMessage(content="Hi")]

            mock_response = {"messages": [AIMessage(content="Ответ")]}
            mock_agent.ainvoke.return_value = mock_response

            # Act
            answer, updated_history = await agent_process_message(
                agent=mock_agent,
                user_message="Вопрос",
                chat_history=chat_history
            )

            # Assert
            assert len(updated_history) == 4  # 2 + 2 новых, меньше лимита 20

    @pytest.mark.asyncio
    async def test_empty_chat_history(self, mock_agent):
        """Тест обработки с пустой историей чата."""
        # Arrange
        user_message = "Первое сообщение"
        chat_history = []

        mock_response = {"messages": [AIMessage(content="Первый ответ")]}
        mock_agent.ainvoke.return_value = mock_response

        # Act
        answer, updated_history = await agent_process_message(
            agent=mock_agent,
            user_message=user_message,
            chat_history=chat_history
        )

        # Assert
        assert answer == "Первый ответ"
        assert len(updated_history) == 2
        assert updated_history[0].content == user_message
        assert updated_history[1].content == "Первый ответ"

    @pytest.mark.asyncio
    async def test_agent_exception_propagated(self, mock_agent):
        """Тест, что исключения от агента пробрасываются."""
        # Arrange
        mock_agent.ainvoke.side_effect = Exception("LLM Error")

        # Act & Assert
        with pytest.raises(Exception, match="LLM Error"):
            await agent_process_message(
                agent=mock_agent,
                user_message="test",
                chat_history=[]
            )


class TestEdgeCases:
    """Тесты для краевых случаев."""

    @pytest.mark.asyncio
    async def test_history_at_exact_limit(self, mock_agent):
        """Тест когда история точно на границе лимита."""
        # Arrange
        with patch("websocket_handler.MAX_CHAT_HISTORY", 2):
            # 4 сообщения = ровно лимит, после добавления будет 6 > 4
            chat_history = [
                HumanMessage(content="1"), AIMessage(content="1"),
                HumanMessage(content="2"), AIMessage(content="2"),
            ]

            mock_response = {"messages": [AIMessage(content="3")]}
            mock_agent.ainvoke.return_value = mock_response

            # Act
            answer, history = await agent_process_message(
                agent=mock_agent,
                user_message="3",
                chat_history=chat_history
            )

            # Assert
            assert len(history) == 4  # обрезано до лимита
            assert history[0].content == "2"  # остались последние 2 пары
