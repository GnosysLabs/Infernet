import { isTauri, invoke } from "@tauri-apps/api/core";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { useCallback, useEffect, useRef, useState } from "react";

type UpdatePhase = "idle" | "available" | "installing" | "error";

export type AppUpdaterState = {
  phase: UpdatePhase;
  version: string | null;
  error: string | null;
  checkNow: () => Promise<void>;
  installAndRestart: () => Promise<void>;
  dismissError: () => void;
};

const FIRST_CHECK_DELAY_MS = 8_000;
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1_000;

export function useAppUpdater(): AppUpdaterState {
  const updateRef = useRef<Update | null>(null);
  const checkingRef = useRef(false);
  const [phase, setPhase] = useState<UpdatePhase>("idle");
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkNow = useCallback(async () => {
    if (!isTauri() || checkingRef.current) return;
    checkingRef.current = true;
    try {
      const update = await check({ timeout: 30_000 });
      if (!update) return;
      if (updateRef.current && updateRef.current !== update) {
        await updateRef.current.close().catch(() => undefined);
      }
      updateRef.current = update;
      setVersion(update.version);
      setError(null);
      setPhase("available");
    } catch (updateError) {
      // Background checks are intentionally quiet. Installation failures are
      // surfaced because the user explicitly asked the app to update.
      console.error("Infernet update check failed", updateError);
    } finally {
      checkingRef.current = false;
    }
  }, []);

  const installAndRestart = useCallback(async () => {
    const update = updateRef.current;
    if (!update || phase === "installing") return;
    setPhase("installing");
    setError(null);
    try {
      await invoke("prepare_for_app_update");
      await update.downloadAndInstall();
      await relaunch();
    } catch (installError) {
      setError(String(installError).replace(/^Error:\s*/i, ""));
      setPhase("error");
    }
  }, [phase]);

  const dismissError = useCallback(() => {
    setError(null);
    setPhase(updateRef.current ? "available" : "idle");
  }, []);

  useEffect(() => {
    if (!isTauri()) return;
    const firstCheck = window.setTimeout(() => void checkNow(), FIRST_CHECK_DELAY_MS);
    const interval = window.setInterval(() => void checkNow(), CHECK_INTERVAL_MS);
    return () => {
      window.clearTimeout(firstCheck);
      window.clearInterval(interval);
    };
  }, [checkNow]);

  return { phase, version, error, checkNow, installAndRestart, dismissError };
}
