import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class DatabaseManager:
    """Управление SQLite базой данных истории вычислений"""

    def __init__(self, db_path: str = "data/history.db"):
        """
        Инициализация менеджера БД

        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Создаёт таблицы если их нет"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Таблица истории вычислений
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS history
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               input_text
                               TEXT
                               NOT
                               NULL,
                               command
                               TEXT,
                               expression
                               TEXT,
                               result_type
                               TEXT
                               NOT
                               NULL,
                               result_text
                               TEXT,
                               result_json
                               TEXT,
                               error_message
                               TEXT,
                               execution_time
                               REAL,
                               tags
                               TEXT,
                               favorite
                               INTEGER
                               DEFAULT
                               0
                           )
                           """)

            # Индексы для быстрого поиска
            cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_timestamp
                               ON history(timestamp DESC)
                           """)

            cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_command
                               ON history(command)
                           """)

            cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_favorite
                               ON history(favorite)
                           """)

            conn.commit()

    def add_entry(self,
                  input_text: str,
                  command: str,
                  expression: str,
                  result_type: str,
                  result_text: str = None,
                  result_json: dict = None,
                  error_message: str = None,
                  execution_time: float = None,
                  tags: List[str] = None) -> int:
        """
        Добавляет запись в историю

        Args:
            input_text: Исходный ввод пользователя
            command: Команда (solve, plot, derivative и т.д.)
            expression: Обработанное выражение
            result_type: Тип результата (success, error, plot_2d и т.д.)
            result_text: Текстовый результат
            result_json: JSON результат (для графиков)
            error_message: Сообщение об ошибке
            execution_time: Время выполнения в секундах
            tags: Теги для категоризации

        Returns:
            ID добавленной записи
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()
            tags_str = json.dumps(tags) if tags else None
            result_json_str = json.dumps(result_json) if result_json else None

            cursor.execute("""
                           INSERT INTO history (timestamp, input_text, command, expression,
                                                result_type, result_text, result_json, error_message,
                                                execution_time, tags)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (
                               timestamp, input_text, command, expression,
                               result_type, result_text, result_json_str, error_message,
                               execution_time, tags_str
                           ))

            conn.commit()
            return cursor.lastrowid

    def get_history(self,
                    limit: int = 100,
                    offset: int = 0,
                    command_filter: str = None,
                    search_query: str = None,
                    favorites_only: bool = False) -> List[Dict]:
        """
        Получает историю вычислений

        Args:
            limit: Количество записей
            offset: Смещение для пагинации
            command_filter: Фильтр по команде
            search_query: Поисковый запрос
            favorites_only: Только избранное

        Returns:
            Список записей истории
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM history WHERE 1=1"
            params = []

            if command_filter:
                query += " AND command = ?"
                params.append(command_filter)

            if search_query:
                query += " AND (input_text LIKE ? OR result_text LIKE ?)"
                search_pattern = f"%{search_query}%"
                params.extend([search_pattern, search_pattern])

            if favorites_only:
                query += " AND favorite = 1"

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_entry(self, entry_id: int) -> Optional[Dict]:
        """Получает одну запись по ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM history WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

            return dict(row) if row else None

    def toggle_favorite(self, entry_id: int) -> bool:
        """Переключает статус избранного"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Получаем текущий статус
            cursor.execute("SELECT favorite FROM history WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

            if not row:
                return False

            new_status = 0 if row[0] else 1

            cursor.execute(
                "UPDATE history SET favorite = ? WHERE id = ?",
                (new_status, entry_id)
            )

            conn.commit()
            return True

    def delete_entry(self, entry_id: int) -> bool:
        """Удаляет запись"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self, keep_favorites: bool = True):
        """Очищает историю"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if keep_favorites:
                cursor.execute("DELETE FROM history WHERE favorite = 0")
            else:
                cursor.execute("DELETE FROM history")

            conn.commit()

    def get_statistics(self) -> Dict:
        """Получает статистику использования"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Общее количество
            cursor.execute("SELECT COUNT(*) FROM history")
            total = cursor.fetchone()[0]

            # По командам
            cursor.execute("""
                           SELECT command, COUNT(*) as count
                           FROM history
                           GROUP BY command
                           ORDER BY count DESC
                           """)
            by_command = dict(cursor.fetchall())

            # Избранное
            cursor.execute("SELECT COUNT(*) FROM history WHERE favorite = 1")
            favorites = cursor.fetchone()[0]

            # Ошибки
            cursor.execute("SELECT COUNT(*) FROM history WHERE result_type = 'error'")
            errors = cursor.fetchone()[0]

            # Средняя скорость
            cursor.execute("""
                           SELECT AVG(execution_time)
                           FROM history
                           WHERE execution_time IS NOT NULL
                           """)
            avg_time = cursor.fetchone()[0] or 0

            return {
                'total': total,
                'by_command': by_command,
                'favorites': favorites,
                'errors': errors,
                'avg_execution_time': round(avg_time, 3)
            }

    def export_to_json(self, filepath: str):
        """Экспортирует историю в JSON"""
        history = self.get_history(limit=10000)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def import_from_json(self, filepath: str):
        """Импортирует историю из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for entry in history:
                # Пропускаем id, он будет создан автоматически
                entry.pop('id', None)

                cursor.execute("""
                               INSERT INTO history (timestamp, input_text, command, expression,
                                                    result_type, result_text, result_json, error_message,
                                                    execution_time, tags, favorite)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                               """, (
                                   entry.get('timestamp'),
                                   entry.get('input_text'),
                                   entry.get('command'),
                                   entry.get('expression'),
                                   entry.get('result_type'),
                                   entry.get('result_text'),
                                   entry.get('result_json'),
                                   entry.get('error_message'),
                                   entry.get('execution_time'),
                                   entry.get('tags'),
                                   entry.get('favorite', 0)
                               ))

            conn.commit()
