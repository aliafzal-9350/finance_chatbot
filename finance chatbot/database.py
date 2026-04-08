"""
MongoDB Database Handler (READ-ONLY + Auth/Admin support)
Industrial-grade connection with error handling and validation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError

from config import Config


class DatabaseHandler:
    AUTH_COLLECTION = "users"

    def __init__(
        self,
        database_name: Optional[str] = None,
        invoice_collection: Optional[str] = None,
        line_items_collection: Optional[str] = None,
    ):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.database_name = database_name or Config.DATABASE_NAME
        self.invoice_collection = invoice_collection or Config.INVOICE_COLLECTION
        self.line_items_collection = line_items_collection or Config.LINE_ITEMS_COLLECTION

    def connect(self) -> bool:
        try:
            try:
                Config.validate()
            except Exception as e:
                print(f"⚠️ Configuration validation warning: {e}")

            self.client = MongoClient(
                Config.MONGO_URI,
                serverSelectionTimeoutMS=10_000,
                connectTimeoutMS=10_000,
                socketTimeoutMS=20_000,
                retryWrites=True,
            )

            self.client.admin.command("ping")
            self.db = self.client[self.database_name]

            print(f"✅ Connected to MongoDB database: {self.database_name}")
            return True

        except ServerSelectionTimeoutError:
            print("❌ MongoDB connection timeout.")
            return False
        except ConnectionFailure as e:
            print(f"❌ Connection failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            print("✅ Database connection closed")

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify_password(self, plain_password: str, stored_hash: str) -> bool:
        return self._hash_password(plain_password) == stored_hash

    def _get_users_collection(self):
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[self.AUTH_COLLECTION]

    def ensure_auth_indexes(self):
        users = self._get_users_collection()
        users.create_index("username", unique=True)

    def bootstrap_default_users(self):
        users = self._get_users_collection()
        now = datetime.utcnow()

        defaults = [
            {
                "username": "admin",
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "assigned_invoice_collection": None,
                "assigned_line_collection": None,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "username": "user_a",
                "password_hash": self._hash_password("usera123"),
                "role": "user",
                "assigned_invoice_collection": "ZATCA_API_INVOICES_ALL",
                "assigned_line_collection": "ZATCA_API_INVOICES_LINES_ALL",
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "username": "user_b",
                "password_hash": self._hash_password("userb123"),
                "role": "user",
                "assigned_invoice_collection": "zatca-pro",
                "assigned_line_collection": "zatca-pro-line",
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            },
        ]

        for u in defaults:
            users.update_one({"username": u["username"]}, {"$setOnInsert": u}, upsert=True)

    def get_user(self, username: str) -> Optional[Dict]:
        return self._get_users_collection().find_one(
            {"username": username, "is_active": True}, {"_id": 0}
        )

    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        assigned_invoice_collection: Optional[str],
        assigned_line_collection: Optional[str],
    ) -> Tuple[bool, str]:
        doc = {
            "username": username.strip(),
            "password_hash": self._hash_password(password),
            "role": role,
            "assigned_invoice_collection": assigned_invoice_collection if role == "user" else None,
            "assigned_line_collection": assigned_line_collection if role == "user" else None,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        try:
            self._get_users_collection().insert_one(doc)
            return True, "User created successfully."
        except DuplicateKeyError:
            return False, "Username already exists."
        except Exception as e:
            return False, f"Failed to create user: {e}"

    def list_users(self) -> List[Dict]:
        return list(
            self._get_users_collection()
            .find({}, {"_id": 0, "password_hash": 0})
            .sort("created_at", -1)
        )

    def update_user_assignment(
        self,
        username: str,
        role: str,
        assigned_invoice_collection: Optional[str],
        assigned_line_collection: Optional[str],
    ) -> Tuple[bool, str]:
        updates = {"role": role, "updated_at": datetime.utcnow()}
        if role == "admin":
            updates["assigned_invoice_collection"] = None
            updates["assigned_line_collection"] = None
        else:
            updates["assigned_invoice_collection"] = assigned_invoice_collection
            updates["assigned_line_collection"] = assigned_line_collection

        result = self._get_users_collection().update_one({"username": username}, {"$set": updates})
        if result.matched_count == 0:
            return False, "User not found."
        return True, "User updated successfully."

    def get_invoices(self, limit: Optional[int] = None, projection: Optional[Dict] = None) -> pd.DataFrame:
        try:
            if self.db is None:
                raise RuntimeError("Database not connected. Call connect() first.")

            collection = self.db[self.invoice_collection]
            cursor = collection.find({}, projection)

            if limit:
                cursor = cursor.limit(limit)

            data = list(cursor)
            if not data:
                print(f"⚠️ No invoices found in collection: {self.invoice_collection}")
                return pd.DataFrame()

            for d in data:
                d.pop("_id", None)

            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df):,} invoices from {self.invoice_collection}")
            return df

        except Exception as e:
            print(f"❌ Error fetching invoices: {e}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        try:
            if self.db is None:
                raise RuntimeError("Database not connected. Call connect() first.")

            return {
                "invoices_count": self.db[self.invoice_collection].count_documents({}),
                "line_items_count": self.db[self.line_items_collection].count_documents({}),
                "database_name": self.database_name,
                "invoice_collection": self.invoice_collection,
                "line_items_collection": self.line_items_collection,
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}