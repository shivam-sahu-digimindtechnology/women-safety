"""
Incident Risk Assessment Engine.
Implements the 6-stage Incident State Machine:
NORMAL -> POSSIBLE_INCIDENT -> CONFIRMED_BY_CONFIGURED_RULES -> ALERT_QUEUED -> ALERT_SENT -> COOLDOWN

Scenarios Handled:
- Scenario A: Female victim attacked / involved in physical struggle.
              Only the identified female victim is highlighted with a BOLD RED bounding box.
              Qualifies for automatic snapshot in outputs/incidents/YYYY-MM-DD/ and emergency notification.
- Scenario B: Male-only fight (Status: FIGHT SUSPECTED). No female victim alert, no red box, NO woman snapshot.
- Scenario C: Woman present calmly without violence (Status: NORMAL). No snapshot.
- Scenario D: Violence detected with unconfirmed female involvement (Status: VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED).
              Warning displayed on HUD, but NO false red box or woman snapshot.
- Scenario E: Gender classification uncertain (Status: GENDER UNCERTAIN).
              SOS signals remain eligible for universal emergency handling.
- Scenario F: Harassment, stalking, or sudden snatching dynamics.

Restricted-area / restricted-zone logic has been completely removed.
"""

import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import config


class IncidentState(Enum):
    NORMAL = "NORMAL"
    POSSIBLE_INCIDENT = "POSSIBLE_INCIDENT"
    CONFIRMED_BY_CONFIGURED_RULES = "CONFIRMED_BY_CONFIGURED_RULES"
    ALERT_QUEUED = "ALERT_QUEUED"
    ALERT_SENT = "ALERT_SENT"
    COOLDOWN = "COOLDOWN"


class RiskAssessmentEngine:
    """
    Central risk assessment engine that combines multi-modal visual cues,
    advances the 6-stage incident state machine, identifies specific female victims,
    and determines whether an event qualifies for snapshot recording and alert dispatching.
    """

    def __init__(self):
        self.state: IncidentState = IncidentState.NORMAL
        self.state_entered_time: float = time.time()
        self.last_alert_time: float = 0.0
        self.current_incident_type: str = "None"
        self.active_signals: List[str] = []
        self._streak_count: int = 0
        self.victim_female_ids: List[int] = []

    def reset(self):
        """Resets engine to initial NORMAL state."""
        self.state = IncidentState.NORMAL
        self.state_entered_time = time.time()
        self.last_alert_time = 0.0
        self.current_incident_type = "None"
        self.active_signals.clear()
        self._streak_count = 0
        self.victim_female_ids.clear()

    def _set_state(self, new_state: IncidentState, reason: str = ""):
        if self.state != new_state:
            old = self.state.value
            self.state = new_state
            self.state_entered_time = time.time()
            if config.DEBUG_MODE:
                print(f"[State Machine] Transition: {old} -> {new_state.value} ({reason})")

    def mark_alert_queued(self, reason: str = "Alert enqueued in background dispatch worker"):
        self._set_state(IncidentState.ALERT_QUEUED, reason)

    def mark_alert_sent(self, reason: str = "Alert successfully dispatched"):
        self.last_alert_time = time.time()
        self._set_state(IncidentState.ALERT_SENT, reason)

    def evaluate_frame(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates signals from a single processed frame, determines specific victim attribution,
        and advances the 6-stage state machine.
        """
        now = time.time()
        active_points = 0
        reasons: List[str] = []
        primary_incident = "None"
        victim_ids: List[int] = []
        is_qualifying_incident = False

        female_ids = signals.get("female_ids", [])
        male_ids = signals.get("male_ids", [])
        unknown_ids = signals.get("unknown_ids", [])
        women_count = len(female_ids)
        men_count = len(male_ids)
        unknown_count = len(unknown_ids)
        total_persons = women_count + men_count + unknown_count

        weights = config.RISK_WEIGHTS

        # 1. SOS Distress Gesture or Manual Trigger (Highest Priority - Universal Emergency)
        sos_detected = signals.get("sos_detected", False)
        if sos_detected:
            active_points += 95
            sos_det = signals.get("sos_details", {})
            desc = sos_det.get("description", "Distress gesture or SOS keypress")
            reasons.append(f"Distress Signal: {desc}")
            primary_incident = "SOS DETECTED"
            is_qualifying_incident = True
            
            sos_pid = sos_det.get("person_id")
            if sos_pid is not None:
                victim_ids.append(sos_pid)
                if sos_pid in female_ids:
                    primary_incident = "POTENTIAL WOMAN SAFETY INCIDENT (SOS)"

        # 2. Violence / Fight Evaluation (from best.pt CNN-GRU)
        violence_detected = signals.get("violence_detected", False)
        fighting_pairs = signals.get("fighting_pairs", [])
        female_attacked = signals.get("female_attacked", False)
        female_attacked_ids = signals.get("female_attacked_ids", [])

        if violence_detected or fighting_pairs or female_attacked:
            v_conf = signals.get("violence_confidence", 0.0)
            
            # Check victim female association
            if female_attacked_ids:
                # Scenario A: Confirmed female victim in fight
                active_points += weights.get("female_attacked", 90)
                victim_ids.extend(female_attacked_ids)
                reasons.append(f"Physical struggle / attack targeting female (ID {female_attacked_ids})")
                primary_incident = "WOMAN SAFETY INCIDENT — EVIDENCE SUPPORTED"
                is_qualifying_incident = True
            elif any(p[0] in female_ids or p[1] in female_ids for p in fighting_pairs):
                # Scenario A: Female directly in fighting pair
                for p in fighting_pairs:
                    if p[0] in female_ids:
                        victim_ids.append(p[0])
                    if p[1] in female_ids:
                        victim_ids.append(p[1])
                active_points += weights.get("female_attacked", 90)
                reasons.append(f"Female directly involved in physical struggle (ID {victim_ids})")
                primary_incident = "WOMAN SAFETY INCIDENT — EVIDENCE SUPPORTED"
                is_qualifying_incident = True
            elif women_count > 0 and violence_detected:
                # Scenario D: Violence detected but female is distant / involvement unconfirmed
                active_points += weights.get("violence_detected", 60)
                reasons.append("Fight detected in scene; female involvement unconfirmed")
                primary_incident = "VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED"
                is_qualifying_incident = False # Do NOT generate woman safety snapshot!
            elif men_count >= 2 and violence_detected:
                # Scenario B: Male-only fight
                active_points += weights.get("violence_detected", 60)
                reasons.append("Physical struggle between male persons (no female victim)")
                primary_incident = "FIGHT SUSPECTED"
                is_qualifying_incident = False # Do NOT generate woman safety snapshot!
            elif unknown_count > 0 and violence_detected:
                # Scenario E: Gender uncertain in fight
                active_points += weights.get("violence_detected", 55)
                reasons.append("Violence detected; gender of participants uncertain")
                primary_incident = "UNKNOWN / INSUFFICIENT EVIDENCE"
                is_qualifying_incident = False
            else:
                active_points += weights.get("violence_detected", 55)
                reasons.append(f"Physical struggle detected (conf={v_conf:.2f})")
                primary_incident = "FIGHT SUSPECTED"
                is_qualifying_incident = False

        # 3. Fall / Collapse Evaluation
        fall_confirmed = signals.get("fall_confirmed", False)
        fall_person_ids = signals.get("fall_person_ids", [])
        if fall_confirmed:
            fall_score = signals.get("fall_score", 0.0)
            active_points += weights.get("fall_confirmed", 80)
            female_falls = [pid for pid in fall_person_ids if pid in female_ids]
            if female_falls:
                victim_ids.extend(female_falls)
                reasons.append(f"Female collapse / fall detected (ID {female_falls})")
                if primary_incident == "None":
                    primary_incident = "WOMAN FALL SUSPECTED"
                    is_qualifying_incident = True
            else:
                reasons.append(f"Person collapse / fall detected (ID {fall_person_ids})")
                if primary_incident == "None":
                    primary_incident = "FALL SUSPECTED"
                    is_qualifying_incident = False # Male fall is not a woman-specific safety incident

        # 4. Spatio-Temporal Snatching Detection
        snatching_detected = signals.get("snatching_detected", False)
        if snatching_detected:
            snatch_det = signals.get("snatching_details", {})
            desc = snatch_det.get("evidence_reason", "Sudden snatching and rapid departure")
            reasons.append(f"Theft / Snatching: {desc}")
            female_snatch = [p[1] for p in snatch_det.get("suspected_pairs", []) if p[1] in female_ids]
            if female_snatch:
                active_points += 85
                victim_ids.extend(female_snatch)
                if primary_incident == "None":
                    primary_incident = "WOMAN SAFETY INCIDENT — EVIDENCE SUPPORTED"
                    is_qualifying_incident = True
            else:
                active_points += 45
                if primary_incident == "None":
                    primary_incident = "SNATCHING SUSPECTED"
                    is_qualifying_incident = False

        # 5. Encirclement / Surrounding (Informational signal; does NOT qualify for emergency dispatch alone)
        female_surrounded = signals.get("female_surrounded", False)
        surrounded_f_ids = signals.get("surrounded_female_ids", [])
        if female_surrounded:
            active_points += 35
            reasons.append(f"Crowd proximity: female in close proximity to multiple individuals (ID {surrounded_f_ids})")
            if primary_incident == "None":
                primary_incident = "POTENTIAL INCIDENT"
                is_qualifying_incident = False

        # 6. Tailing / Stalking Trajectory
        female_followed = signals.get("female_followed", False)
        followed_f_ids = signals.get("followed_female_ids", [])
        if female_followed:
            active_points += 30
            reasons.append(f"Sustained tailing of female person (ID {followed_f_ids})")
            if primary_incident == "None":
                primary_incident = "POTENTIAL INCIDENT"
                is_qualifying_incident = False

        # Deduplicate victim IDs
        self.victim_female_ids = list(dict.fromkeys(victim_ids))

        # Clamp total score between 0 and 100
        final_score = max(0, min(100, active_points))

        # Determine Risk Level
        if final_score >= config.RISK_THRESHOLD_HIGH:
            level = "HIGH"
        elif final_score >= config.RISK_THRESHOLD_MEDIUM:
            level = "MEDIUM"
        elif final_score >= config.RISK_THRESHOLD_LOW:
            level = "LOW"
        else:
            level = "NORMAL"

        # --- 6-STAGE STATE MACHINE ADVANCEMENT ---
        has_threat_signal = (
            final_score >= config.RISK_THRESHOLD_HIGH or
            (len(reasons) > 0 and final_score >= config.RISK_THRESHOLD_MEDIUM)
        )

        if self.state == IncidentState.NORMAL:
            if has_threat_signal:
                self._streak_count = 1
                self._set_state(IncidentState.POSSIBLE_INCIDENT, "Initial threat signals detected")
                self.current_incident_type = primary_incident

        elif self.state == IncidentState.POSSIBLE_INCIDENT:
            if has_threat_signal:
                self._streak_count += 1
                if self._streak_count >= 2 or final_score >= config.RISK_THRESHOLD_HIGH or sos_detected:
                    self._set_state(IncidentState.CONFIRMED_BY_CONFIGURED_RULES, "Corroborated across frames / high risk")
                    self.current_incident_type = primary_incident
            else:
                self._streak_count = 0
                self._set_state(IncidentState.NORMAL, "Threat signals ceased")

        elif self.state == IncidentState.CONFIRMED_BY_CONFIGURED_RULES:
            # Stays in CONFIRMED_BY_CONFIGURED_RULES until caller enqueues alert
            pass

        elif self.state == IncidentState.ALERT_QUEUED:
            if now - self.state_entered_time > 0.8:
                self.mark_alert_sent("Queued alert proceeded to sent")

        elif self.state == IncidentState.ALERT_SENT:
            self.last_alert_time = now
            self._set_state(IncidentState.COOLDOWN, "Alert dispatched; cooldown initiated")

        elif self.state == IncidentState.COOLDOWN:
            elapsed = now - self.last_alert_time
            if elapsed >= config.ALERT_COOLDOWN_SECONDS:
                self._set_state(IncidentState.NORMAL, "Cooldown period expired")
                self._streak_count = 0

        # Mark whether action/alert and snapshot is worthy in this frame
        # Strictly require qualifying woman-safety incident or verified SOS
        alert_worthy = (
            self.state == IncidentState.CONFIRMED_BY_CONFIGURED_RULES and
            is_qualifying_incident
        )

        # Woman Status Determination for HUD
        if len(self.victim_female_ids) > 0 and is_qualifying_incident:
            is_woman_safe = False
            woman_status_text = "WOMAN SAFETY INCIDENT — EVIDENCE SUPPORTED"
            woman_badge_color = (0, 0, 255) # Red
        elif primary_incident == "VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED":
            is_woman_safe = True
            woman_status_text = "VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED"
            woman_badge_color = (0, 165, 255) # Amber warning
        elif primary_incident == "FIGHT SUSPECTED":
            is_woman_safe = True
            woman_status_text = "FIGHT SUSPECTED"
            woman_badge_color = (0, 140, 255) # Blue
        elif primary_incident == "UNKNOWN / INSUFFICIENT EVIDENCE":
            is_woman_safe = True
            woman_status_text = "UNKNOWN / INSUFFICIENT EVIDENCE"
            woman_badge_color = (160, 160, 160)
        elif women_count == 0:
            is_woman_safe = True
            woman_status_text = "NO FEMALE IN FRAME"
            woman_badge_color = (160, 160, 160)
        elif self.state == IncidentState.POSSIBLE_INCIDENT and is_qualifying_incident:
            is_woman_safe = False
            woman_status_text = "POTENTIAL INCIDENT (VERIFYING)"
            woman_badge_color = (0, 140, 255)
        else:
            is_woman_safe = True
            woman_status_text = "WOMAN STATUS: SECURE (SAFE)"
            woman_badge_color = (46, 204, 113) # Green

        if not reasons or primary_incident == "None" or final_score == 0:
            if not reasons:
                reasons.append("Monitoring active - safe scene")
            if primary_incident == "None":
                primary_incident = "SAFE / NO INCIDENT EVIDENCE"

        summary = f"State: {self.state.value} | Status: {primary_incident} | Risk: {final_score}/100"

        return {
            "score": final_score,
            "level": level,
            "state": self.state.value,
            "status_label": primary_incident,
            "is_woman_safe": is_woman_safe,
            "woman_status_text": woman_status_text,
            "woman_badge_color": woman_badge_color,
            "victim_female_ids": self.victim_female_ids,
            "is_qualifying_incident": is_qualifying_incident,
            "women_count": women_count,
            "men_count": men_count,
            "unknown_count": unknown_count,
            "total_persons": total_persons,
            "reasons": reasons,
            "primary_incident": primary_incident,
            "alert_worthy": alert_worthy,
            "summary": summary
        }
