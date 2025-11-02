import os
import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque, Dict, List, TypedDict, Any, Optional
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
from openai import OpenAI
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()
API_TOKEN = os.getenv('BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_CREDENTIALS_FILE = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
GOOGLE_TOKEN_FILE = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')
EVENTS_FILE = "events.json"
HISTORY_MAX_MESSAGES = 20

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

class ChatMessage(TypedDict):
    role: str
    content: str

# In-memory per-chat history: chat_id -> deque of ChatMessage
chat_histories: Dict[int, Deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=HISTORY_MAX_MESSAGES))

SYSTEM_PROMPT_BASE = (
    "You are an assistant that manages user's events. Events can be stored in a local JSON file (storage_type='json') "
    "or in Google Calendar (storage_type='calendar'/'google'/'google_calendar'). "
    "When the user asks to create, list, get, update, or delete events, use the provided tools. "
    "ALL CRUD operations require a 'storage_type' parameter to specify where to store/retrieve events. "
    "IMPORTANT: When the user mentions an event by NAME (not event_id) for Google Calendar, you MUST first use search_events "
    "with storage_type='calendar' to find the event and get its event_id. Only then use update_event or delete_event "
    "with the correct event_id and storage_type. Do NOT use event names as event_id - always search first when given a name. "
    "Prefer concise replies. Dates should be ISO (YYYY-MM-DD) and times 24h HH:MM when possible. "
)

def get_system_prompt() -> str:
    now = datetime.now().astimezone()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    tz_str = now.tzname() or "local time"
    return (
        SYSTEM_PROMPT_BASE
        + f"Today is {date_str}, current time is {time_str} {tz_str}. "
        + "Interpret relative dates like 'today', 'tomorrow', 'next Monday' relative to this date/time."
    )

# -------------------------
# Event Storage Interface (SOLID - Interface Segregation)
# -------------------------
class EventStorage(ABC):
    """Abstract interface for event storage operations."""
    
    @abstractmethod
    async def create_event(self, title: str, date: str, time: str, description: str) -> Dict[str, Any]:
        """Create a new event."""
        pass
    
    @abstractmethod
    async def list_events(self) -> List[Dict[str, Any]]:
        """List all events."""
        pass
    
    @abstractmethod
    async def get_event(self, event_id: Any) -> Dict[str, Any]:
        """Get an event by ID."""
        pass
    
    @abstractmethod
    async def update_event(self, event_id: Any, updates: Dict[str, str]) -> Dict[str, Any]:
        """Update an event by ID."""
        pass
    
    @abstractmethod
    async def delete_event(self, event_id: Any) -> bool:
        """Delete an event by ID."""
        pass

# -------------------------
# JSON File Event Storage Implementation
# -------------------------
class JSONFileEventStorage(EventStorage):
    """Implementation of EventStorage using JSON file."""
    
    def __init__(self, file_path: str = EVENTS_FILE):
        self.file_path = file_path
    
    async def _read_json_file(self) -> Dict[str, Any]:
        """Read events from JSON file."""
        def _read() -> Dict[str, Any]:
            if not os.path.exists(self.file_path):
                return {"events": []}
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {"events": []}
        return await asyncio.to_thread(_read)
    
    async def _write_json_file(self, data: Dict[str, Any]) -> None:
        """Write events to JSON file."""
        def _write() -> None:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        await asyncio.to_thread(_write)
    
    def _get_next_id(self, events: List[Dict[str, Any]]) -> int:
        """Generate next event ID."""
        if not events:
            return 1
        return max(int(e.get("id", 0)) for e in events) + 1
    
    async def create_event(self, title: str, date: str, time: str, description: str) -> Dict[str, Any]:
        """Create a new event in JSON file."""
        data = await self._read_json_file()
        events: List[Dict[str, Any]] = data.get("events", [])
        event_id = self._get_next_id(events)
        event = {
            "id": int(event_id),
            "title": title or "",
            "date": date or "",
            "time": time or "",
            "description": description or "",
        }
        events.append(event)
        data["events"] = events
        await self._write_json_file(data)
        return event
    
    async def list_events(self) -> List[Dict[str, Any]]:
        """List all events from JSON file."""
        data = await self._read_json_file()
        return list(data.get("events", []))
    
    async def get_event(self, event_id: Any) -> Dict[str, Any]:
        """Get an event by ID from JSON file."""
        data = await self._read_json_file()
        events: List[Dict[str, Any]] = data.get("events", [])
        for e in events:
            if int(e.get("id")) == int(event_id):
                return e
        raise ValueError("Event not found")
    
    async def update_event(self, event_id: Any, updates: Dict[str, str]) -> Dict[str, Any]:
        """Update an event by ID in JSON file."""
        data = await self._read_json_file()
        events: List[Dict[str, Any]] = data.get("events", [])
        for e in events:
            if int(e.get("id")) == int(event_id):
                for k in ["title", "date", "time", "description"]:
                    if k in updates and updates[k] is not None:
                        e[k] = updates[k]
                await self._write_json_file({"events": events})
                return e
        raise ValueError("Event not found")
    
    async def delete_event(self, event_id: Any) -> bool:
        """Delete an event by ID from JSON file."""
        data = await self._read_json_file()
        events: List[Dict[str, Any]] = data.get("events", [])
        new_events = [e for e in events if int(e.get("id")) != int(event_id)]
        if len(new_events) == len(events):
            raise ValueError("Event not found")
        await self._write_json_file({"events": new_events})
        return True
    
    async def list_events_text(self) -> str:
        """Format events as text for display."""
        events = await self.list_events()
        if not events:
            return "No events found."
        lines = []
        for e in sorted(events, key=lambda x: (x.get("date", ""), x.get("time", ""))):
            lines.append(f"#{e.get('id')} | {e.get('date','')} {e.get('time','')} | {e.get('title','')} | {e.get('description','')}")
        return "\n".join(lines)

# -------------------------
# Google Calendar Event Storage Implementation
# -------------------------
class GoogleCalendarEventStorage(EventStorage):
    """Implementation of EventStorage using Google Calendar."""
    
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    
    def __init__(self, credentials_file: str = GOOGLE_CREDENTIALS_FILE, 
                 token_file: str = GOOGLE_TOKEN_FILE):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self._service: Optional[Any] = None
    
    def _get_service(self) -> Any:
        """Get authenticated Google Calendar service."""
        if self._service:
            return self._service
        
        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing Google credentials: {e}")
                    raise ValueError("Google Calendar service not available. Check credentials.")
            else:
                if not os.path.exists(self.credentials_file):
                    raise ValueError(f"Google credentials file not found: {self.credentials_file}")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        try:
            self._service = build('calendar', 'v3', credentials=creds)
            return self._service
        except Exception as e:
            raise ValueError(f"Error building Google Calendar service: {e}")
    
    def _parse_datetime(self, date: str, time: str) -> datetime:
        """Parse date and time into datetime object."""
        try:
            if date:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if time:
                    time_parts = time.split(':')
                    hour = int(time_parts[0])
                    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                    return date_obj.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    return date_obj.replace(hour=9, minute=0, second=0, microsecond=0)
            else:
                return datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        except Exception:
            return datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    
    async def create_event(self, title: str, date: str, time: str, description: str, 
                          duration_hours: float = 1.0) -> Dict[str, Any]:
        """Create an event in Google Calendar."""
        service = self._get_service()
        
        def _create() -> Dict[str, Any]:
            start_datetime = self._parse_datetime(date, time)
            end_datetime = start_datetime + timedelta(hours=duration_hours)
            
            event = {
                'summary': title or "Untitled Event",
                'description': description or "",
                'start': {
                    'dateTime': start_datetime.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_datetime.isoformat(),
                    'timeZone': 'UTC',
                },
            }
            
            created = service.events().insert(calendarId='primary', body=event).execute()
            return {
                "id": created.get('id'),
                "htmlLink": created.get('htmlLink'),
                "summary": created.get('summary'),
                "start": created.get('start', {}).get('dateTime'),
                "end": created.get('end', {}).get('dateTime'),
            }
        
        return await asyncio.to_thread(_create)
    
    async def list_events(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """List upcoming events from Google Calendar."""
        service = self._get_service()
        
        def _list() -> List[Dict[str, Any]]:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            return [
                {
                    "id": e.get('id'),
                    "summary": e.get('summary', 'No Title'),
                    "start": e.get('start', {}).get('dateTime') or e.get('start', {}).get('date'),
                    "end": e.get('end', {}).get('dateTime') or e.get('end', {}).get('date'),
                    "description": e.get('description', ''),
                }
                for e in events
            ]
        
        return await asyncio.to_thread(_list)
    
    async def get_event(self, event_id: Any) -> Dict[str, Any]:
        """Get a specific Google Calendar event by ID."""
        service = self._get_service()
        
        def _get() -> Dict[str, Any]:
            event = service.events().get(calendarId='primary', eventId=str(event_id)).execute()
            return {
                "id": event.get('id'),
                "summary": event.get('summary', 'No Title'),
                "start": event.get('start', {}).get('dateTime') or event.get('start', {}).get('date'),
                "end": event.get('end', {}).get('dateTime') or event.get('end', {}).get('date'),
                "description": event.get('description', ''),
                "htmlLink": event.get('htmlLink'),
            }
        
        return await asyncio.to_thread(_get)
    
    async def update_event(self, event_id: Any, updates: Dict[str, str]) -> Dict[str, Any]:
        """Update a Google Calendar event."""
        service = self._get_service()
        
        def _update() -> Dict[str, Any]:
            event = service.events().get(calendarId='primary', eventId=str(event_id)).execute()
            
            if "title" in updates and updates["title"]:
                event['summary'] = updates["title"]
            if "description" in updates and updates["description"]:
                event['description'] = updates["description"]
            
            # Handle date/time updates
            date = updates.get("date")
            time = updates.get("time")
            
            if date is not None or time is not None:
                try:
                    # Parse current event datetime if it exists
                    if 'dateTime' in event.get('start', {}):
                        old_start_str = event['start']['dateTime'].replace('Z', '+00:00')
                        old_end_str = event['end']['dateTime'].replace('Z', '+00:00')
                        old_start = datetime.fromisoformat(old_start_str)
                        old_end = datetime.fromisoformat(old_end_str)
                        duration = old_end - old_start
                        
                        # Build new start datetime
                        if date and time:
                            # Both provided - use new date and time
                            start_dt = self._parse_datetime(date, time)
                        elif date:
                            # Only date provided - keep old time, use new date
                            start_dt = old_start.replace(year=datetime.strptime(date, "%Y-%m-%d").year,
                                                         month=datetime.strptime(date, "%Y-%m-%d").month,
                                                         day=datetime.strptime(date, "%Y-%m-%d").day)
                        elif time:
                            # Only time provided - keep old date, use new time
                            time_parts = time.split(':')
                            hour = int(time_parts[0])
                            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                            start_dt = old_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        else:
                            start_dt = old_start
                        
                        # Calculate new end time preserving duration
                        end_dt = start_dt + duration
                        event['start']['dateTime'] = start_dt.isoformat()
                        event['end']['dateTime'] = end_dt.isoformat()
                    elif 'date' in event.get('start', {}):
                        # All-day event - handle differently if needed
                        if date:
                            event['start']['date'] = date
                            event['end']['date'] = date
                except Exception as e:
                    print(f"Error updating event datetime: {e}")
                    import traceback
                    traceback.print_exc()
            
            updated = service.events().update(calendarId='primary', eventId=str(event_id), body=event).execute()
            return {
                "id": updated.get('id'),
                "summary": updated.get('summary'),
                "start": updated.get('start', {}).get('dateTime') or updated.get('start', {}).get('date'),
                "end": updated.get('end', {}).get('dateTime') or updated.get('end', {}).get('date'),
            }
        
        return await asyncio.to_thread(_update)
    
    async def delete_event(self, event_id: Any) -> bool:
        """Delete a Google Calendar event."""
        service = self._get_service()
        
        def _delete() -> bool:
            service.events().delete(calendarId='primary', eventId=str(event_id)).execute()
            return True
        
        return await asyncio.to_thread(_delete)
    
    async def search_events_by_name(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Google Calendar events by name/summary."""
        service = self._get_service()
        
        def _search() -> List[Dict[str, Any]]:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now,
                q=query,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            return [
                {
                    "id": e.get('id'),
                    "summary": e.get('summary', 'No Title'),
                    "start": e.get('start', {}).get('dateTime') or e.get('start', {}).get('date'),
                    "end": e.get('end', {}).get('dateTime') or e.get('end', {}).get('date'),
                    "description": e.get('description', ''),
                }
                for e in events
            ]
        
        return await asyncio.to_thread(_search)

# -------------------------
# Storage Instances (Dependency Injection)
# -------------------------
json_storage = JSONFileEventStorage()
calendar_storage = GoogleCalendarEventStorage()

# Storage type mapping
STORAGE_MAP = {
    "json": json_storage,
    "calendar": calendar_storage,
    "google": calendar_storage,
    "google_calendar": calendar_storage,
}

# -------------------------
# Unified CRUD Functions (work with any EventStorage)
# -------------------------
async def create_event_unified(storage: EventStorage, title: str, date: str, time: str, 
                                description: str, **kwargs) -> Dict[str, Any]:
    """Create event using specified storage."""
    if isinstance(storage, GoogleCalendarEventStorage):
        duration_hours = kwargs.get("duration_hours", 1.0)
        return await storage.create_event(title, date, time, description, duration_hours)
    else:
        return await storage.create_event(title, date, time, description)

async def update_event_unified(storage: EventStorage, event_id: Any, 
                               updates: Dict[str, str]) -> Dict[str, Any]:
    """Update event using specified storage."""
    return await storage.update_event(event_id, updates)

async def delete_event_unified(storage: EventStorage, event_id: Any) -> bool:
    """Delete event using specified storage."""
    return await storage.delete_event(event_id)

async def get_event_unified(storage: EventStorage, event_id: Any) -> Dict[str, Any]:
    """Get event using specified storage."""
    return await storage.get_event(event_id)

async def list_events_unified(storage: EventStorage, **kwargs) -> List[Dict[str, Any]]:
    """List events using specified storage."""
    if isinstance(storage, GoogleCalendarEventStorage):
        max_results = kwargs.get("max_results", 10)
        return await storage.list_events(max_results)
    else:
        return await storage.list_events()

def get_storage_by_type(storage_type: str) -> EventStorage:
    """Get storage instance by type name."""
    storage_type_lower = storage_type.lower()
    if storage_type_lower not in STORAGE_MAP:
        raise ValueError(f"Unknown storage type: {storage_type}. Available: {list(STORAGE_MAP.keys())}")
    return STORAGE_MAP[storage_type_lower]

# -------------------------
# OpenAI tools (function calling)
# -------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a new calendar event. Specify storage_type as 'json' for local file or 'calendar'/'google' for Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["json", "calendar", "google", "google_calendar"], "description": "Storage type: 'json' for local file, 'calendar'/'google'/'google_calendar' for Google Calendar."},
                    "title": {"type": "string", "description": "Short name of the event."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD if known, else empty."},
                    "time": {"type": "string", "description": "Time in HH:MM 24h if known, else empty."},
                    "description": {"type": "string", "description": "Optional longer description."},
                    "duration_hours": {"type": "number", "description": "Duration in hours (default 1.0, only for calendar storage)."},
                },
                "required": ["storage_type", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_event",
            "description": "Update fields of an existing event by id. Specify storage_type. For calendar: if user mentions event by NAME, first use search_events to find event_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["json", "calendar", "google", "google_calendar"], "description": "Storage type: 'json' for local file, 'calendar'/'google'/'google_calendar' for Google Calendar."},
                    "event_id": {"type": ["string", "integer"], "description": "Event ID (integer for json, string for calendar)."},
                    "title": {"type": "string", "description": "New event title."},
                    "date": {"type": "string", "description": "New date in YYYY-MM-DD."},
                    "time": {"type": "string", "description": "New time in HH:MM 24h."},
                    "description": {"type": "string", "description": "New description."},
                },
                "required": ["storage_type", "event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete an event by id. Specify storage_type. For calendar: if user mentions event by NAME, first use search_events to find event_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["json", "calendar", "google", "google_calendar"], "description": "Storage type: 'json' for local file, 'calendar'/'google'/'google_calendar' for Google Calendar."},
                    "event_id": {"type": ["string", "integer"], "description": "Event ID (integer for json, string for calendar)."},
                },
                "required": ["storage_type", "event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_event",
            "description": "Get an event by id. Specify storage_type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["json", "calendar", "google", "google_calendar"], "description": "Storage type: 'json' for local file, 'calendar'/'google'/'google_calendar' for Google Calendar."},
                    "event_id": {"type": ["string", "integer"], "description": "Event ID (integer for json, string for calendar)."},
                },
                "required": ["storage_type", "event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List all events. Specify storage_type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["json", "calendar", "google", "google_calendar"], "description": "Storage type: 'json' for local file, 'calendar'/'google'/'google_calendar' for Google Calendar."},
                    "max_results": {"type": "integer", "description": "Maximum number of results (only for calendar storage, default 10)."},
                },
                "required": ["storage_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Search events by name/summary. Only available for Google Calendar storage. Use this FIRST when user mentions an event by name. ALWAYS follow up by calling update_event or delete_event with the event_id from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "storage_type": {"type": "string", "enum": ["calendar", "google", "google_calendar"], "description": "Storage type must be calendar/google/google_calendar."},
                    "query": {"type": "string", "description": "Event name or part of event name to search for."},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default 10)."}
                },
                "required": ["storage_type", "query"]
            }
        }
    },
]

async def call_openai_with_tools(messages: List[ChatMessage], max_iterations: int = 5) -> str:
    """Recursively handle tool calls until LLM returns a final response."""
    if max_iterations <= 0:
        return "Maximum tool call iterations reached. Please try again."

    def _api_call() -> Any:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )

    response = await asyncio.to_thread(_api_call)
    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []

    # If the model asked to call tools, execute them and send results back
    tool_result_messages: List[Dict[str, Any]] = []
    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = {}
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except Exception:
                args = {}

            try:
                # Get storage type and storage instance
                storage_type = args.get("storage_type", "")
                storage = get_storage_by_type(storage_type) if storage_type else None
                
                if name == "create_event":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    else:
                        result = await create_event_unified(
                            storage=storage,
                            title=args.get("title", ""),
                            date=args.get("date", ""),
                            time=args.get("time", ""),
                            description=args.get("description", ""),
                            duration_hours=args.get("duration_hours", 1.0),
                        )
                elif name == "update_event":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    else:
                        # Build updates dict - include all provided keys (even if None/empty)
                        updates = {
                            k: args.get(k)
                            for k in ["title", "date", "time", "description"]
                            if k in args
                        }
                        
                        event_id = args.get("event_id")
                        # Convert event_id based on storage type
                        if storage_type.lower() in ["json"]:
                            event_id = int(event_id)
                        else:
                            # Ensure event_id is string for Google Calendar
                            event_id = str(event_id)
                        result = await update_event_unified(storage, event_id, updates)
                elif name == "delete_event":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    else:
                        event_id = args.get("event_id")
                        # Convert event_id based on storage type
                        if storage_type.lower() in ["json"]:
                            event_id = int(event_id)
                        await delete_event_unified(storage, event_id)
                        result = {"ok": True}
                elif name == "get_event":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    else:
                        event_id = args.get("event_id")
                        # Convert event_id based on storage type
                        if storage_type.lower() in ["json"]:
                            event_id = int(event_id)
                        result = await get_event_unified(storage, event_id)
                elif name == "list_events":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    else:
                        result = await list_events_unified(
                            storage=storage,
                            max_results=args.get("max_results", 10)
                        )
                elif name == "search_events":
                    if not storage:
                        result = {"error": "storage_type is required"}
                    elif storage_type.lower() not in ["calendar", "google", "google_calendar"]:
                        result = {"error": "search_events is only available for Google Calendar storage"}
                    else:
                        # This method is specific to GoogleCalendarEventStorage
                        if isinstance(storage, GoogleCalendarEventStorage):
                            result = await storage.search_events_by_name(
                                query=args.get("query", ""),
                                max_results=args.get("max_results", 10)
                            )
                        else:
                            result = {"error": "search_events is only available for Google Calendar storage"}
                else:
                    result = {"error": f"Unknown tool {name}"}
            except Exception as e:
                result = {"error": str(e)}

            tool_result_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        follow_up: List[ChatMessage] = messages + [
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                # attach tool_calls for compliance
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            }
        ] + tool_result_messages  # type: ignore[arg-type]

        # Recursively continue if more tool calls are needed
        return await call_openai_with_tools(follow_up, max_iterations - 1)

    # No tool calls - return the final response
    return (assistant_message.content or "").strip()

@dp.message(Command(commands=["start"]))
async def cmd_start(message: Message):
    await message.answer("Hello! I can manage your events. Use /help to see options.")

@dp.message(Command(commands=["help"]))
async def cmd_help(message: Message):
    await message.answer(
        "You can say things like: 'Book an event for tomorrow morning',\n"
        "'Delete an event with name \"Breakfast\" from 3rd of June', or use commands:\n"
        "/list - list events, /reset - clear chat context."
    )

@dp.message(Command(commands=["list"]))
async def cmd_list(message: Message):
    await message.answer(await json_storage.list_events_text())

@dp.message(Command(commands=["reset"]))
async def cmd_reset(message: Message):
    chat_histories.pop(message.chat.id, None)
    await message.answer("Context has been reset.")

@dp.message()
async def reply_with_llm(message: Message):
    user_text = message.text or ""

    # Build messages: system + history + current user
    history = chat_histories[message.chat.id]
    messages: List[ChatMessage] = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_text})

    try:
        await message.chat.do("typing")
    except Exception:
        pass

    try:
        llm_reply = await call_openai_with_tools(messages)
        if not llm_reply:
            llm_reply = "(No response from model)"

        # Update history (store user and assistant turns)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": llm_reply})

        await message.answer(llm_reply)
    except Exception as e:
        print(f"Error: {e}")
        await message.answer("Sorry, I couldn't reach the LLM right now. Please try again later.")

if __name__ == "__main__":
    dp.run_polling(bot)
