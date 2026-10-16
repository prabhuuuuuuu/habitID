import React, { createContext, useContext, useState, useRef } from 'react';
import { BehavioralCollector } from '../services/behavioralCollector';
import { emptySignals } from '../models/BehavioralSignal';
import type { BehavioralSignals } from '../models/BehavioralSignal';
import type { HabitIDResult } from '../models/HabitIDResult';

interface AppState {
  signals: BehavioralSignals;
  scoreResult: HabitIDResult | null;
  collector: BehavioralCollector;
  setSignals: (s: Partial<BehavioralSignals>) => void;
  setScoreResult: (r: HabitIDResult) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [signals, setSignalsState] = useState<BehavioralSignals>(emptySignals());
  const [scoreResult, setScoreResult] = useState<HabitIDResult | null>(null);
  const collectorRef = useRef(new BehavioralCollector());

  const setSignals = (partial: Partial<BehavioralSignals>) => {
    setSignalsState((prev) => ({ ...prev, ...partial }));
  };

  return (
    <AppContext.Provider
      value={{
        signals,
        scoreResult,
        collector: collectorRef.current,
        setSignals,
        setScoreResult,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside AppProvider');
  return ctx;
}
