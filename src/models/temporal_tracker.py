
class RuleBasedTemporalTracker:
    """
    A deterministic tracker that replaces the black-box Bi-LSTM.
    It analyzes a sliding window of recent CMAG/Fusion scores to detect 
    Spikes, Rising Trends, and Sustained Aggression.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, current_score: float) -> dict:
        """
        Updates the tracker with a new chunk score and returns the latest trend analysis.
        """
        self.history.append(current_score)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        return self._analyze_window()
        
    def _analyze_window(self) -> dict:
        if len(self.history) == 0:
            return {"trend": "stable", "escalation_score": 0.0, "prediction": "stable"}
            
        current = self.history[-1]
        previous = self.history[-2] if len(self.history) > 1 else current
        
        is_spike = current > 0.45 and previous < 0.4
        is_sustained = sum(1 for s in self.history if s > 0.5) >= 3
        is_falling = current < self.history[0] - 0.2
        
        is_rising = False
        if len(self.history) >= 3:
            # Check if generally increasing
            # Not strictly monotonic, but end is significantly higher than start 
            # and it isn't wildly fluctuating down
            if current - self.history[0] > 0.15:
                drops = sum(1 for i in range(len(self.history)-1) if self.history[i] > self.history[i+1])
                if drops <= 1:
                    is_rising = True
                    
        escalation_score = 0.0
        prediction = "stable"
        trend = "stable"
        
        # Priority rules
        if is_spike:
            trend = "spike"
            # Spike gives a strong +0.2 boost on top of current, capped at 0.95. No flatline 0.9 anymore.
            escalation_score = min(current + 0.2, 0.95)
            prediction = "violence likely to escalate"
        elif is_falling:
            trend = "falling"
            escalation_score = max(0.0, current)
            prediction = "situation de-escalating"
        elif is_rising:
            trend = "rising"
            # Smooth blend of current and highest warning, allowing it to organically fluctuate
            escalation_score = (current * 0.7) + (max(self.history) * 0.3)
            prediction = "violence likely to escalate"
        elif is_sustained:
            trend = "sustained"
            # Prioritize current value to let the UI breathe, but keep a 20% memory of the peak
            escalation_score = (current * 0.8) + (max(self.history) * 0.2)
            prediction = "ongoing violence"
        else:
            escalation_score = current
            
        return {
            "trend": trend,
            "escalation_score": escalation_score,
            "prediction": prediction
        }
