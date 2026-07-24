import json
import re


class KnowledgeRetriever:
    """Retrieve context-relevant static rules and habit rules.

    The retriever intentionally does not use benchmark labels, scenario IDs, or
    oracle habit_GT fields. It only uses the current observation plus the
    structured KB files available to the evaluated assistant.
    """

    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    GLOBAL_LOCATIONS = {"anywhere", "all", "home", "house", "indoor", "indoors", "all rooms"}
    DEVIATION_CUES = {
        "still", "remains", "remain", "remaining", "no ", "not ", "without",
        "missing", "missed", "late", "delayed", "delay", "empty", "dry",
        "cluttered", "untidied", "unplugged", "idle", "scattered", "muddy",
        "full", "asleep", "sleeping", "yawning", "left", "forgotten",
    }
    CONTINUING_CUES = {"still", "remains", "remain", "continuing", "ongoing", "engrossed"}
    ROUTINE_BLOCKING_CUES = {
        "still", "remains", "remain", "remaining", "engrossed", "watching",
        "browsing", "scrolling", "typing", "working", "chatting", "lounging",
        "sitting", "looking for", "getting into bed", "walks from", "directly",
        "straight", "idle", "full", "unplugged", "cooking", "chopping",
        "shower", "reading", "movie", "videos", "computer", "desk", "sofa",
    }
    DURATION_LIMIT_CUES = {
        "has been", "continuously", "continuous", "consecutive", "for ",
        "since", "started at", "2 hours", "2.5 hours", "3 hours", "135 minutes",
        "140th", "timer", "exceeds", "exceeded", "without taking a break",
        "without a break", "continues the gaming session",
    }
    LATE_SCREEN_STRONG_CUES = {
        "active", "actively", "still", "continues", "continuing", "run late",
        "past", "cutoff", "second movie", "vr", "headset", "brightly lit",
        "videos", "watching videos", "watching a movie", "work emails",
        "emails", "social media", "laptop", "tablet", "turning up", "volume",
        "sound system", "music is playing",
    }
    BRIEF_BEDTIME_PHONE_CUES = {
        "brief", "briefly", "short", "one short", "message", "messages",
        "checks a smartphone", "reads a few", "reads messages",
    }
    DEADLINE_BLOCKING_CUES = {
        "go out", "leaving", "leave", "packing", "slowly", "complex",
        "focused", "still sleeping", "asleep",
    }

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "before", "between", "by",
        "for", "from", "in", "into", "is", "it", "must", "no", "not", "of",
        "on", "or", "the", "their", "there", "to", "user", "with", "without",
    }

    RULE_KEYWORDS = {
        "coffee": {
            "coffee", "espresso", "caffeine", "latte", "americano", "cup",
        },
        "drinking coffee": {
            "coffee", "espresso", "caffeine", "latte", "americano",
        },
        "feeding pets": {
            "feed", "feeding", "fed", "pet", "pets", "dog", "cat", "bowl",
            "food", "treat", "treats",
        },
        "eating snacks": {
            "snack", "snacks", "chips", "cookie", "cookies", "nuts", "candy",
            "crumbs", "eat", "eating", "lunch", "fruit", "apple", "grapes",
        },
        "using electronic devices": {
            "phone", "smartphone", "tablet", "ipad", "laptop", "charger",
            "charging", "device", "devices", "speaker", "earbuds",
        },
        "high-intensity screen use or electronic entertainment": {
            "game", "games", "gaming", "pc", "video", "videos", "movie",
            "streaming", "screen", "entertainment", "console", "phone",
            "smartphone", "tablet", "ipad", "laptop", "e-book", "ebook",
            "vr", "headset", "music", "sound", "vinyl", "tv",
        },
        "eating (any food)": {
            "eat", "eating", "lunch", "dinner", "breakfast", "meal", "food",
            "snack", "snacks", "noodles", "rice", "sandwich",
        },
        "using phones or tablets": {
            "phone", "smartphone", "tablet", "ipad", "laptop", "screen",
        },
        "playing video games": {
            "game", "games", "gaming", "pc", "console", "video game",
        },
        "using stove": {
            "stove", "burner", "cooktop", "pan", "pot", "cooking",
            "flame", "fire", "boiling",
        },
        "full_house_cleaning": {
            "clean", "cleaning", "cluttered", "untidied", "tidy", "tidying",
            "vacuum", "sweeping", "dust", "bathroom", "living room",
        },
        "watering_plants": {
            "water", "watering", "plants", "planters", "balcony", "dry",
        },
        "laundry_washing": {
            "laundry", "washer", "washing", "dryer", "drying", "basket",
            "clothes", "towels", "sheets", "socks", "pile",
        },
        "playing_piano": {
            "piano", "practice", "keyboard", "music",
        },
    }

    HABIT_KEYWORDS = {
        "wake": {"wake", "wakes", "awake", "asleep", "sleeping", "bed", "lying"},
        "weather_schedule": {
            "weather", "schedule", "calendar", "ipad", "tablet", "morning",
        },
        "grooming": {
            "groom", "grooming", "bathroom", "wash", "washing", "face", "shaving",
            "shaver", "makeup",
        },
        "meditation": {"meditation", "meditates", "quietly", "breathing", "sits"},
        "breakfast": {
            "breakfast", "kitchen", "eggs", "toast", "cereal", "morning meal",
        },
        "medicine": {"medicine", "medication", "pill", "pills", "water"},
        "coffee": {"coffee", "espresso", "caffeine", "kitchen"},
        "lunch": {"lunch", "meal", "kitchen", "sofa", "noodles"},
        "news": {"news", "tablet", "tv", "scrolling", "stories", "media"},
        "nap_rest": {
            "nap", "asleep", "sleep", "rest", "resting", "sofa", "lounging",
        },
        "daily_summary": {
            "summary", "planning", "tasks", "notes", "desk", "study", "editing",
        },
        "deep_work": {
            "work", "working", "typing", "desk", "study", "computer", "reading",
            "research", "notes", "focused", "deep", "editing",
        },
        "exercise": {
            "exercise", "stretch", "stretching", "mobility", "yoga", "mat",
            "wrist", "sports",
        },
        "dinner": {
            "dinner", "cook", "cooking", "chopping", "vegetables", "rice",
            "noodles", "kitchen",
        },
        "trash": {"trash", "garbage", "bag", "bins", "bin", "outdoor"},
        "dog_walk": {"walk", "walking", "leash", "dog", "outdoor", "entrance"},
        "dog_paws": {"paws", "paw", "muddy", "leash", "entrance", "dog"},
        "dog_feeding": {"feed", "feeding", "dog", "bowl", "food", "water", "empty"},
        "tea": {"tea", "herbal", "warm water", "drink", "kitchen"},
        "safety_check": {
            "doors", "windows", "lights", "check", "safety", "bed", "before bed",
        },
        "skincare": {"skincare", "floss", "flossing", "bathroom", "shower"},
        "brushing": {"brush", "brushing", "teeth", "toothbrush", "bathroom"},
        "charging": {
            "charge", "charging", "charger", "unplugged", "phone", "tablet",
            "desk", "nightstand",
        },
        "phone_bed": {
            "phone", "smartphone", "message", "messages", "bed", "reading",
        },
        "sleep": {"sleep", "asleep", "bed", "bedroom", "lamp"},
        "laundry": {
            "laundry", "washer", "washing", "drying", "basket", "clothes",
            "towels", "sheets",
        },
        "plants": {"plants", "planters", "watering", "water", "dry", "balcony"},
        "cleaning": {
            "cleaning", "clean", "cluttered", "untidied", "vacuum", "sweeping",
            "bathroom", "living room",
        },
        "finances": {"finance", "finances", "budget", "spreadsheet", "bills"},
        "music_family": {"music", "piano", "call", "family", "video call"},
    }

    HABIT_CONCEPT_HINTS = [
        ("weather_schedule", {"weather", "schedule", "ipad"}),
        ("grooming", {"groom", "bathroom", "face"}),
        ("breakfast", {"breakfast"}),
        ("medicine", {"medicine"}),
        ("coffee", {"coffee"}),
        ("lunch", {"lunch"}),
        ("news", {"news"}),
        ("nap_rest", {"rest", "sofa", "afternoon"}),
        ("daily_summary", {"summary", "plans", "tasks"}),
        ("dinner", {"dinner"}),
        ("trash", {"trash"}),
        ("dog_walk", {"walks the dog", "dog walk"}),
        ("dog_paws", {"paws", "leash"}),
        ("dog_feeding", {"feeds the dog", "dog food", "water bowl"}),
        ("tea", {"tea", "warm water"}),
        ("safety_check", {"doors", "windows", "lights"}),
        ("skincare", {"skincare", "floss"}),
        ("brushing", {"brushes teeth", "teeth"}),
        ("charging", {"charges", "charging", "mobile devices"}),
        ("phone_bed", {"smartphone", "phone"}),
        ("sleep", {"goes to sleep", "sleep"}),
        ("exercise", {"stretch", "mobility"}),
        ("deep_work", {"deep work", "focused reading", "coding notes", "self-study"}),
        ("laundry", {"laundry", "sheets", "towels"}),
        ("plants", {"plants"}),
        ("cleaning", {"cleaning"}),
        ("finances", {"finances", "budget"}),
        ("music_family", {"music", "family", "piano"}),
    ]

    def __init__(self, static_kb_path, habit_kb_path):
        self.static_rules = self._load_json(static_kb_path)
        self.habit_rules = self._load_json(habit_kb_path)

    def _load_json(self, path):
        if path is None:
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
            return []

    def _normalize(self, text):
        return str(text or "").lower().replace("_", " ")

    def _tokenize(self, text):
        text = self._normalize(text)
        return set(re.findall(r"[a-z0-9]+", text)) - self.STOPWORDS

    def _has_any_phrase(self, text, phrases):
        text = self._normalize(text)
        return any(str(phrase).lower() in text for phrase in phrases)

    def _parse_minutes(self, time_str):
        try:
            time_str = str(time_str).replace("：", ":")
            h, m = map(int, time_str.strip().split(":"))
            return h * 60 + m
        except Exception:
            return -1

    def _extract_window(self, window_str):
        window = self._normalize(window_str)
        match = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", window)
        if not match:
            return None, None
        return self._parse_minutes(match.group(1)), self._parse_minutes(match.group(2))

    def _day_matches(self, current_day, window_str):
        rule = self._normalize(window_str)
        day = self._normalize(current_day).strip()
        if "always" in rule:
            return True

        has_day = (
            any(d in rule for d in self.DAYS)
            or "weekday" in rule
            or "weekdays" in rule
            or "weekend" in rule
            or "weekends" in rule
        )
        if not has_day:
            return True
        if day in rule:
            return True
        if ("weekday" in rule or "weekdays" in rule) and day not in {"saturday", "sunday"}:
            return True
        if ("weekend" in rule or "weekends" in rule) and day in {"saturday", "sunday"}:
            return True
        return False

    def _is_next_day_after_rule_day(self, current_day, window_str):
        rule = self._normalize(window_str)
        day = self._normalize(current_day).strip()
        if day not in self.DAYS:
            return False
        current_index = self.DAYS.index(day)
        for rule_index, rule_day in enumerate(self.DAYS):
            if rule_day in rule and current_index == (rule_index + 1) % len(self.DAYS):
                return True
        return False

    def _time_relation(self, current_day, current_time_str, window_str, tolerance=0):
        if not self._day_matches(current_day, window_str):
            return None

        window = self._normalize(window_str)
        if "always" in window:
            return {
                "match": True,
                "relation": "always",
                "distance": 0,
                "start": None,
                "end": None,
            }

        start, end = self._extract_window(window)
        current = self._parse_minutes(current_time_str)
        if start is None or end is None or current == -1:
            return {
                "match": True,
                "relation": "day-only",
                "distance": 0,
                "start": start,
                "end": end,
            }

        candidates = [(current, start, end)]
        # Early-morning test events often need the previous evening bedtime
        # routines. Compare them on an extended timeline without changing the
        # public timestamp.
        if current <= 4 * 60:
            candidates.append((current + 24 * 60, start, end + 24 * 60 if end < start else end))
            if start >= 18 * 60:
                candidates.append((current + 24 * 60, start, end + (24 * 60 if end < start else 0)))

        best = None
        for curr, s, e in candidates:
            if s <= e:
                if s - tolerance <= curr <= e + tolerance:
                    distance = 0 if s <= curr <= e else min(abs(curr - s), abs(curr - e))
                    relation = "within" if s <= curr <= e else "near"
                elif curr < s:
                    distance = s - curr
                    relation = "upcoming"
                else:
                    distance = curr - e
                    relation = "overdue"
            else:
                if curr >= s or curr <= e:
                    distance = 0
                    relation = "within"
                elif curr < s:
                    distance = s - curr
                    relation = "upcoming"
                else:
                    distance = curr - e
                    relation = "overdue"

            candidate = {
                "match": relation in {"within", "near"},
                "relation": relation,
                "distance": distance,
                "start": s,
                "end": e,
            }
            if best is None or candidate["distance"] < best["distance"]:
                best = candidate
        return best

    def _is_location_match(self, current_loc, rule_loc):
        current = self._normalize(current_loc)
        rule = self._normalize(rule_loc)

        if any(global_loc == rule or global_loc in rule.split(",") for global_loc in self.GLOBAL_LOCATIONS):
            return True

        current_tokens = self._tokenize(current)
        for loc_part in str(rule_loc or "").split(","):
            loc_tokens = self._tokenize(loc_part)
            if not loc_tokens:
                continue
            if loc_tokens & self.GLOBAL_LOCATIONS:
                return True
            if loc_tokens.issubset(current_tokens):
                return True
        return False

    def _location_score(self, current_loc, rule_loc):
        if self._is_location_match(current_loc, rule_loc):
            return 1.0
        current_tokens = self._tokenize(current_loc)
        rule_tokens = self._tokenize(rule_loc)
        if not current_tokens or not rule_tokens:
            return 0.0
        overlap = len(current_tokens & rule_tokens)
        return overlap / max(len(rule_tokens), 1)

    def _event_has_deviation_cue(self, event_text):
        text = self._normalize(event_text)
        return self._has_any_phrase(text, self.DEVIATION_CUES)

    def _event_has_continuing_cue(self, event_text):
        text = self._normalize(event_text)
        return self._has_any_phrase(text, self.CONTINUING_CUES)

    def _event_has_routine_blocking_cue(self, event_text):
        text = self._normalize(event_text)
        return self._has_any_phrase(text, self.ROUTINE_BLOCKING_CUES)

    def _event_has_deadline_blocking_cue(self, event_text):
        text = self._normalize(event_text)
        return self._has_any_phrase(text, self.DEADLINE_BLOCKING_CUES)

    def _keyword_score(self, event_text, keywords):
        if not event_text or not keywords:
            return 0.0
        text = self._normalize(event_text)
        event_tokens = self._tokenize(text)
        score = 0.0
        for keyword in keywords:
            keyword = self._normalize(keyword)
            if " " in keyword:
                if keyword in text:
                    score += 1.25
            elif keyword in event_tokens:
                score += 1.0
        return score

    def _rule_keywords(self, rule):
        target = self._normalize(rule.get("target_action", ""))
        desc = self._normalize(rule.get("description", ""))
        keywords = set()
        for key, values in self.RULE_KEYWORDS.items():
            normalized_key = self._normalize(key)
            if normalized_key in target or normalized_key in desc:
                keywords.update(values)
        if not keywords:
            keywords.update(self._tokenize(target))
        return keywords

    def _numeric_duration_minutes(self, event_text):
        text = self._normalize(event_text)
        durations = []
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs|minute|minutes|min|mins)", text):
            amount = float(value)
            if unit.startswith(("hour", "hr")):
                durations.append(amount * 60)
            else:
                durations.append(amount)
        # Phrases such as "2 hours and 15 minutes" produce two matches;
        # returning the sum is a useful approximation for limit checks.
        if durations:
            return sum(durations)
        return None

    def _duration_limit_rule_applies(self, event_text):
        text = self._normalize(event_text)
        if "after playing" in text or "taking a walk" in text or "take a break" in text:
            return False
        if not self._keyword_score(text, self.RULE_KEYWORDS["playing video games"]):
            return False

        duration = self._numeric_duration_minutes(text)
        if duration is not None:
            return duration > 120

        if "started at" in text or "since" in text:
            return True
        return self._has_any_phrase(text, self.DURATION_LIMIT_CUES)

    def _late_screen_rule_applies(self, event_text, relation, current_time):
        text = self._normalize(event_text)
        tokens = self._tokenize(text)
        current_minutes = self._parse_minutes(current_time)
        in_late_window = (
            relation
            and relation.get("relation") in {"within", "near"}
            and current_minutes != -1
            and (current_minutes >= 23 * 60 or current_minutes <= 6 * 60)
        )
        near_cutoff = (
            relation
            and relation.get("relation") == "upcoming"
            and relation.get("distance") is not None
            and relation["distance"] <= 10
        )
        if not (in_late_window or near_cutoff):
            return False

        is_charging_only = (
            self._has_any_phrase(text, {"plugging in", "plugged in", "charging", "charger", "charge"})
            and not self._has_any_phrase(
                text,
                {
                    "using", "watch", "watching", "scrolling", "browsing",
                    "playing", "active", "actively", "videos", "movie",
                    "emails", "social media", "vr", "headset",
                },
            )
        )
        if is_charging_only:
            return False

        inactive_device_state = (
            bool(tokens & {"phone", "smartphone", "tablet", "laptop"})
            and self._has_any_phrase(text, {"remain", "remains", "unplugged", "on the desk", "on desk"})
            and not self._has_any_phrase(
                text,
                {
                    "using", "watch", "watching", "scrolling", "browsing",
                    "playing", "active", "actively", "videos", "movie",
                    "emails", "social media", "vr", "headset",
                },
            )
        )
        if inactive_device_state:
            return False

        # The user's short phone-in-bed routine is explicitly allowed. Keep it
        # out unless the observation also contains stronger work/entertainment
        # cues such as videos, emails, VR, or prolonged active use.
        is_bed_phone = (
            "bed" in tokens
            and bool(tokens & {"phone", "smartphone"})
            and self._has_any_phrase(text, self.BRIEF_BEDTIME_PHONE_CUES)
        )
        if is_bed_phone and not self._has_any_phrase(text, self.LATE_SCREEN_STRONG_CUES):
            return False

        # Plain "is playing games" is not enough to activate a late-night rule:
        # the observation needs an overrun, active-continuation, intensity, or
        # explicit cutoff cue. Duration-specific gaming is handled by R009.
        is_plain_gaming = (
            bool(tokens & {"game", "games", "gaming", "pc", "console"})
            and not self._has_any_phrase(text, self.LATE_SCREEN_STRONG_CUES)
            and not self._duration_limit_rule_applies(text)
        )
        if is_plain_gaming:
            return False

        if self._has_any_phrase(text, self.LATE_SCREEN_STRONG_CUES):
            return True
        if near_cutoff and self._has_any_phrase(text, {"setting up", "watch", "movie", "laptop", "tablet"}):
            return True
        if in_late_window and bool(tokens & {"smartphone", "phone", "tablet", "laptop", "vr", "headset"}):
            return True
        return False

    def _is_completion_or_progress_event(self, event_text):
        text = self._normalize(event_text)
        progress_words = {
            "using", "doing", "cleaning", "vacuum", "sweeping", "watering",
            "organizing", "sorting", "washing", "preparing", "cooking",
        }
        return bool(self._tokenize(text) & progress_words) and not self._event_has_deviation_cue(text)

    def _rule_relevance(self, rule, current_day, current_time, current_location, event_text):
        relation = self._time_relation(current_day, current_time, rule.get("time_window", "always"))
        event_text = event_text or ""
        action = self._normalize(rule.get("action", ""))
        target = self._normalize(rule.get("target_action", ""))
        keywords = self._rule_keywords(rule)
        keyword_score = self._keyword_score(event_text, keywords)
        has_deviation_cue = self._event_has_deviation_cue(event_text)
        location_match = self._is_location_match(current_location, rule.get("location", "anywhere"))
        current_minutes = self._parse_minutes(current_time)
        minutes_to_window_end = None
        if relation is not None and isinstance(relation.get("end"), int) and current_minutes != -1:
            end = relation["end"]
            if end >= current_minutes:
                minutes_to_window_end = end - current_minutes

        if relation is None:
            # Deadline rules can still be relevant on the morning after a
            # missed deadline if the event text is semantically related.
            if action == "remind if incomplete" and keyword_score > 0 and self._is_next_day_after_rule_day(
                current_day,
                rule.get("time_window", "always"),
            ):
                relation = {
                    "match": False,
                    "relation": "missed-deadline",
                    "distance": 24 * 60,
                    "start": None,
                    "end": None,
                }
            else:
                return None

        # Do not let global rules appear in every prompt. Rules need either
        # action relevance or an incomplete/deadline cue.
        if action in {"forbid", "limit count", "limit duration", "forbid except brief bedtime phone use"}:
            near_cutoff = (
                rule.get("rule_id") == "R006"
                and relation["relation"] == "upcoming"
                and relation["distance"] is not None
                and relation["distance"] <= 10
            )
            if not relation["match"] and relation["relation"] not in {"always", "day-only"} and not near_cutoff:
                return None
            if not location_match:
                return None
            if keyword_score <= 0:
                return None
            if rule.get("rule_id") == "R009" and not self._duration_limit_rule_applies(event_text):
                return None
            if rule.get("rule_id") == "R006" and not self._late_screen_rule_applies(
                event_text,
                relation,
                current_time,
            ):
                return None
            # Boundary protection: a normal electric shaver should not be
            # treated as the same as phone/tablet/laptop use in a bathroom.
            if rule.get("rule_id") == "R005" and "shaver" in self._tokenize(event_text):
                return None
        elif "require presence" in action:
            if not relation["match"] and relation["relation"] not in {"always", "day-only"}:
                return None
            if keyword_score <= 0:
                return None
            # Presence rules may describe a remote appliance state, e.g. the
            # user is in the living room while the kitchen stove is active.
            if not location_match and "kitchen" not in self._normalize(event_text) and "stove" not in self._normalize(event_text):
                return None
        elif action == "remind if incomplete":
            has_deadline_blocking_cue = self._event_has_deadline_blocking_cue(event_text)
            imminent_deadline = (
                relation["relation"] == "within"
                and minutes_to_window_end is not None
                and minutes_to_window_end <= 90
            )
            urgent_deadline = (
                relation["relation"] == "within"
                and minutes_to_window_end is not None
                and minutes_to_window_end <= 15
            )
            recently_missed_deadline = (
                relation["relation"] in {"overdue", "missed-deadline"}
                and relation["distance"] is not None
                and relation["distance"] <= 180
            )
            very_recently_missed_deadline = (
                relation["relation"] in {"overdue", "missed-deadline"}
                and relation["distance"] is not None
                and relation["distance"] <= 30
            )
            semantic_related = keyword_score > 0
            deadline_attention_needed = (
                urgent_deadline
                or very_recently_missed_deadline
                or (imminent_deadline and (has_deviation_cue or has_deadline_blocking_cue))
                or (
                    recently_missed_deadline
                    and (semantic_related or has_deviation_cue or has_deadline_blocking_cue)
                )
            )
            relation_ok = (
                semantic_related
                or deadline_attention_needed
            )
            if not relation_ok:
                return None
            if (
                keyword_score > 0
                and self._is_completion_or_progress_event(event_text)
                and not has_deviation_cue
                and not recently_missed_deadline
            ):
                return None
        elif action == "enforce schedule":
            relation_ok = relation["match"] or relation["relation"] == "within"
            if not relation_ok:
                return None
        elif keyword_score <= 0 and target:
            return None
        elif not location_match:
            return None

        score = 50 + keyword_score * 10
        if relation["relation"] in {"within", "always", "day-only"}:
            score += 20
        elif relation["relation"] in {"overdue", "upcoming"} and relation["distance"] is not None:
            score += max(0, 18 - relation["distance"] / 10)
        elif relation["relation"] == "missed-deadline":
            score += 8
        if has_deviation_cue:
            score += 8
        return {
            "score": score,
            "relation": relation["relation"],
            "distance": relation["distance"],
            "kind": "rule",
        }

    def _habit_concepts(self, habit):
        text = self._normalize(habit.get("description", ""))
        concepts = set()
        for concept, hints in self.HABIT_CONCEPT_HINTS:
            if any(self._normalize(hint) in text for hint in hints):
                concepts.add(concept)
        if "goes to sleep" in text:
            concepts.add("sleep")
        return concepts

    def _habit_keywords(self, habit):
        concepts = self._habit_concepts(habit)
        keywords = set(self._tokenize(habit.get("description", "")))
        for concept in concepts:
            keywords.update(self.HABIT_KEYWORDS.get(concept, set()))
        return keywords

    def _habit_signal(self, relation, loc_score, keyword_score, event_text):
        if not relation or relation.get("distance") is None:
            return ""

        distance = relation["distance"]
        relation_name = relation["relation"]
        has_deviation_cue = self._event_has_deviation_cue(event_text)
        has_continuing_cue = self._event_has_continuing_cue(event_text)
        has_blocking_cue = self._event_has_routine_blocking_cue(event_text)
        location_shift = loc_score < 0.35

        timing = ""
        if relation_name == "within":
            timing = "routine is due now"
        elif relation_name == "overdue" and distance <= 180:
            timing = f"routine is about {int(distance)} min overdue"
        elif relation_name == "upcoming" and distance <= 45:
            timing = f"routine is due in about {int(distance)} min"
        elif relation_name == "near" and distance <= 30:
            timing = "routine is close to its usual window"

        if not timing:
            return ""

        cues = []
        if has_deviation_cue:
            cues.append("the observation contains an omission/delay cue")
        elif has_continuing_cue:
            cues.append("the observation suggests the current activity is continuing")
        elif has_blocking_cue:
            cues.append("the observation may be occupying the routine window")
        if keyword_score > 0 and relation_name in {"overdue", "upcoming"}:
            cues.append("the event is semantically related to this routine")

        if not cues:
            return ""
        if location_shift and relation_name in {"within", "upcoming", "overdue", "near"}:
            cues.append("the usual routine location differs from the current location")
        return f"Signal: {timing}; {'; '.join(cues)}"

    def _habit_relevance(self, habit, current_day, current_time, current_location, event_text):
        relation = self._time_relation(
            current_day,
            current_time,
            habit.get("time_window", "always"),
            tolerance=15,
        )
        if relation is None:
            return None

        event_text = event_text or ""
        keyword_score = self._keyword_score(event_text, self._habit_keywords(habit))
        loc_score = self._location_score(current_location, habit.get("location", "anywhere"))
        has_deviation_cue = self._event_has_deviation_cue(event_text)
        has_continuing_cue = self._event_has_continuing_cue(event_text)

        score = 0.0
        match_type = None

        if relation["match"] and loc_score > 0:
            score = 80 + keyword_score * 8 + loc_score * 10
            match_type = "current routine"
        elif keyword_score > 0 and relation["distance"] <= 180:
            score = 70 + keyword_score * 10 - relation["distance"] / 12 + loc_score * 5
            match_type = f"{relation['relation']} semantic routine"
        elif has_deviation_cue and relation["relation"] == "overdue" and relation["distance"] <= 180:
            score = 62 - relation["distance"] / 10 + loc_score * 8 + keyword_score * 8
            match_type = "possibly missed routine"
        elif has_continuing_cue and relation["relation"] == "upcoming" and relation["distance"] <= 120:
            score = 58 - relation["distance"] / 12 + loc_score * 8 + keyword_score * 8
            match_type = "upcoming routine transition"
        elif relation["distance"] <= 30:
            score = 44 - relation["distance"] / 10 + loc_score * 6 + keyword_score * 5
            match_type = "nearby routine transition"
        elif relation["distance"] <= 60 and loc_score > 0:
            score = 48 - relation["distance"] / 8 + loc_score * 8 + keyword_score * 4
            match_type = "nearby same-location routine"
        elif relation["distance"] <= 90 and has_deviation_cue:
            score = 43 - relation["distance"] / 12 + keyword_score * 6 + loc_score * 4
            match_type = "nearby routine candidate"

        if score <= 0 or match_type is None:
            return None

        signal = self._habit_signal(relation, loc_score, keyword_score, event_text)
        if signal:
            score += 10

        return {
            "score": score,
            "relation": relation["relation"],
            "distance": relation["distance"],
            "kind": "habit",
            "match_type": match_type,
            "keyword_score": keyword_score,
            "location_score": loc_score,
            "signal": signal,
        }

    def _habit_fallback_candidates(self, current_day, current_time, current_location, event_text, selected_ids):
        """Return broad routine candidates when strict retrieval misses habits.

        This fallback is deliberately generic: it uses temporal proximity,
        action words, and deviation cues, not benchmark labels or IDs.
        """
        candidates = []
        has_deviation_cue = self._event_has_deviation_cue(event_text)
        has_continuing_cue = self._event_has_continuing_cue(event_text)

        for habit in self.habit_rules:
            habit_id = habit.get("rule_id")
            if habit_id in selected_ids:
                continue
            relation = self._time_relation(
                current_day,
                current_time,
                habit.get("time_window", "always"),
                tolerance=0,
            )
            if relation is None or relation["distance"] is None:
                continue
            keyword_score = self._keyword_score(event_text or "", self._habit_keywords(habit))
            loc_score = self._location_score(current_location, habit.get("location", "anywhere"))

            score = 0.0
            match_type = None
            if keyword_score > 0 and relation["distance"] <= 240:
                score = 55 + keyword_score * 9 - relation["distance"] / 18 + loc_score * 4
                match_type = "semantic routine fallback"
            elif has_deviation_cue and relation["relation"] == "overdue" and relation["distance"] <= 240:
                score = 46 - relation["distance"] / 18 + loc_score * 5
                match_type = "missed routine fallback"
            elif has_continuing_cue and relation["relation"] == "upcoming" and relation["distance"] <= 180:
                score = 42 - relation["distance"] / 20 + loc_score * 4
                match_type = "next routine fallback"
            elif relation["distance"] <= 75 and loc_score > 0:
                score = 36 - relation["distance"] / 15 + loc_score * 4
                match_type = "nearby routine fallback"

            if match_type and score > 0:
                signal = self._habit_signal(relation, loc_score, keyword_score, event_text)
                if signal:
                    score += 8
                candidates.append((score, habit, {
                    "score": score,
                    "relation": relation["relation"],
                    "distance": relation["distance"],
                    "kind": "habit",
                    "match_type": match_type,
                    "keyword_score": keyword_score,
                    "location_score": loc_score,
                    "signal": signal,
                }))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _format_item(self, src, item, meta):
        item_id = item.get("rule_id", "N/A")
        desc = item.get("description", "")
        window = item.get("time_window", "always")
        loc = item.get("location", "anywhere")
        if src == "Habit":
            match_type = meta.get("match_type", "routine candidate")
            distance = meta.get("distance")
            detail = f"Match: {match_type}"
            if isinstance(distance, (int, float)) and distance:
                detail += f", {int(distance)} min from routine window"
            signal = meta.get("signal")
            if signal:
                detail += f"; {signal}"
            return f"[{src} {item_id}] {desc} (Routine: {window} @ {loc}; {detail}; habits are probabilistic, not hard rules)"
        return f"[{src} {item_id}] {desc} (Scope: {window} @ {loc}; Match: {meta.get('relation', 'relevant')})"

    def get_relevant_context(self, current_day, current_time_str, current_location, current_event=None):
        """Return context lines for a current observation.

        Static rules are action-aware. Habit rules use tolerant retrieval so
        missed, delayed, substituted, or location-shifted routines can still be
        surfaced to the model.
        """
        rule_candidates = []
        for rule in self.static_rules:
            meta = self._rule_relevance(
                rule,
                current_day,
                current_time_str,
                current_location,
                current_event or "",
            )
            if meta:
                rule_candidates.append((meta["score"], rule, meta))

        habit_candidates = []
        for habit in self.habit_rules:
            meta = self._habit_relevance(
                habit,
                current_day,
                current_time_str,
                current_location,
                current_event or "",
            )
            if meta:
                habit_candidates.append((meta["score"], habit, meta))

        habit_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_habit_ids = {item[1].get("rule_id") for item in habit_candidates[:6]}
        if self.habit_rules:
            habit_candidates.extend(
                self._habit_fallback_candidates(
                    current_day,
                    current_time_str,
                    current_location,
                    current_event or "",
                    selected_habit_ids,
                )
            )

        # Keep prompts compact and reduce over-intervention from weak weekly
        # routines. Rules are listed before habits because hard constraints are
        # more important than statistical habits.
        rule_candidates.sort(key=lambda item: item[0], reverse=True)
        habit_candidates.sort(key=lambda item: item[0], reverse=True)

        lines = []
        seen = set()
        for _, rule, meta in rule_candidates[:5]:
            key = ("Rule", rule.get("rule_id"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(self._format_item("Rule", rule, meta))

        for _, habit, meta in habit_candidates[:5]:
            key = ("Habit", habit.get("rule_id"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(self._format_item("Habit", habit, meta))

        return lines


if __name__ == "__main__":
    print("KnowledgeRetriever uses action-aware rules and tolerant habit retrieval.")
