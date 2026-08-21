/**
 * Notebook toolkit for the tutoring session.
 *
 * Gives the tutor agent small, high-level tools (nb_add_cell, nb_edit_cell,
 * nb_delete_cell, nb_read, nb_run) instead of raw bash + marimo code-mode
 * boilerplate. The extension generates the `cm` ceremony itself, which:
 *   - cuts token usage (the model sends only the cell body),
 *   - removes a whole class of errors observed in real sessions:
 *       cold kernel (cells never run)      -> warm-up call before every op
 *       redundant mo/nx/np/plt imports     -> stripped automatically
 *       editing a nonexistent cell         -> pre-check, returns cell list
 *   - keeps the student's terminal quiet: each call renders as one friendly
 *     status line ("📝 Setting up your first question…") with output hidden
 *     behind the expand keybinding. The LLM still receives full output.
 *
 * Upload boxes carry a 📨 Send to my tutor button: pressing it appends to
 * session_artifacts/student_signal.txt, which the watcher below turns into a
 * turn, so the tutor never has to ask whether the photo is up. Everything
 * else the student explores (sliders, widgets) has no button — they say so
 * in the terminal and the tutor reads the values with nb_read.
 */
import { spawn, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";
import { uuidv7 } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai/compat";
import { Text } from "@earendil-works/pi-tui";
import { keyHint, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** This file's directory. */
const EXT_DIR = path.dirname(fileURLToPath(import.meta.url));

/** JSON string literals are valid Python string literals. */
const py = (s: string) => JSON.stringify(s);

/**
 * The argument of an `mo.md(...)` call, as a RAW triple-quoted literal.
 *
 * `py()` is wrong here, and silently so. marimo recognises a markdown cell
 * when it saves the file and rewrites it as a triple-quoted literal, copying
 * the r/f prefix from the ORIGINAL source token — so a note created as
 * `mo.md("…\\frac…")` is written back as a NON-raw `mo.md("""…\frac…""")`,
 * and the next reload lets Python eat `\f`, `\a`, `\r`, `\t`. That is how
 * the module's central definition ended up on screen as
 * `C_i = rac{ ext{friendships…}}`, and $L = 7/6 pprox 1.17$ beside it.
 * Every note skeleton is full of LaTeX, so this hit the keepsake hardest.
 *
 * A raw string cannot contain `"""` and cannot end in a backslash, so the
 * text is split into raw chunks glued together with ordinary literals. In
 * the normal case (no triple quote anywhere) that is exactly the
 * `r"""…"""` marimo writes for itself.
 *
 * The prefix marimo copies is the LAST string token's, so that token must be
 * raw no matter what the text ends with — and a note ends with the student's
 * own words, which can perfectly well end in a quotation mark. Padding the
 * final chunk with one newline (invisible in markdown) keeps it raw instead
 * of peeling it into an ordinary literal and losing the prefix again.
 */
function pyMd(markdown: string): string {
  const chunk = (s: string): string => {
    if (!s) return '""';
    // Neither a quote nor a backslash may sit against the closing """ —
    // peel any trailing run of them into an ordinary escaped literal.
    const tail = /["\\]+$/.exec(s);
    if (!tail) return `r"""${s}"""`;
    const head = s.slice(0, s.length - tail[0].length);
    return head ? `r"""${head}""" ${py(tail[0])}` : py(tail[0]);
  };
  const parts = markdown.replace(/\r\n/g, "\n").split('"""');
  const last = parts.length - 1;
  if (parts[last] === "" || /["\\]$/.test(parts[last])) parts[last] += "\n";
  return parts.map(chunk).join(` '"""' `);
}
const pyList = (xs: string[]) => JSON.stringify(xs);
const sanitize = (s: string) => s.replace(/\W/g, "_");

/**
 * The starter notebook already owns mo/nx/np/plt. Models add these imports
 * anyway, which triggers marimo's multiply-defined-name rejection (seen in
 * production) — strip them from submitted cell bodies.
 */
function stripRedundantImports(code: string): string {
  const redundant = [
    /^\s*import marimo as mo\s*$/,
    /^\s*import marimo\s*$/,
    /^\s*import networkx as nx\s*$/,
    /^\s*import numpy as np\s*$/,
    /^\s*import matplotlib\.pyplot as plt\s*$/,
    /^\s*from matplotlib import pyplot as plt\s*$/,
    /^\s*import igraph as ig\s*$/,
    /^\s*import seaborn as sns\s*$/,
    /^\s*import altair as alt\s*$/,
    /^\s*import pandas as pd\s*$/,
  ];
  return code
    .split("\n")
    .filter((line) => !redundant.some((re) => re.test(line)))
    .join("\n");
}

/**
 * Ask the browser to bring a cell into view (marimo's focus-cell op) — new
 * content should greet the student, not hide below the fold. Wrapped in
 * try/except so a marimo-internals change can never fail the operation.
 */
const focusCellCode = (cellIdExpr: string, indent: string) =>
  `${indent}try:\n` +
  `${indent}    from marimo._messaging.notification import FocusCellNotification as _FCN\n` +
  `${indent}    ctx.broadcast_raw_notification(_FCN(cell_id=${cellIdExpr}))\n` +
  `${indent}except Exception:\n` +
  `${indent}    pass\n`;

/**
 * Python source of the improvised-cell review (nb_review.py), prepended to
 * the kernel call that creates the cell. Missing file = review skipped.
 */
let reviewSrcCache: string | null = null;
function reviewSource(): string {
  if (reviewSrcCache === null) {
    // Beside this file inside the package; the cwd path is the pre-package
    // layout, kept so an older module folder still reviews its cells.
    const candidates = [
      path.join(EXT_DIR, "nb_review.py"),
      path.join(process.cwd(), ".pi", "extensions", "nb_review.py"),
    ];
    reviewSrcCache = "";
    for (const c of candidates) {
      try {
        reviewSrcCache = fs.readFileSync(c, "utf-8");
        break;
      } catch {
        /* try the next one */
      }
    }
  }
  return reviewSrcCache ?? "";
}

const indentBlock = (s: string, n: number) =>
  s
    .split("\n")
    .map((l) => (l.trim() ? " ".repeat(n) + l : l))
    .join("\n");

const BOOTSTRAP =
  `import marimo._code_mode as cm\n` +
  `async with cm.get_context() as ctx:\n` +
  `    for _c in list(ctx.cells):\n` +
  `        ctx.run_cell(_c.id)\n`;

// ---------------------------------------------------------------------------
// Talking to the marimo kernel.
//
// Two HTTP calls, and that is the whole protocol:
//   GET  /api/sessions          -> { "<session id>": { path, filename }, ... }
//   POST /api/kernel/execute    -> an SSE stream of stdout / stderr / done
//
// This used to be marimo-pair's execute-code.sh, spawned through bash for every
// nb_* call. That cost a bash + curl process per call and, worse, put `jq` and
// `curl` on the list of things a student has to install — on Windows, where
// Git Bash ships curl but not jq, that was the single most likely thing to
// stop a first evening dead. Node speaks HTTP; nothing here needs a shell.
// ---------------------------------------------------------------------------

const KERNEL_TIMEOUT_MS = 180_000;
/** How long a nb_* call waits for the student's notebook page to attach. */
const SESSION_WAIT_MS = 90_000;

// ---------------------------------------------------------------------------
// The notebook server.
//
// This used to be a launcher script: start marimo, poll its log for a URL,
// export MARIMO_URL, open a browser, start pi, kill marimo on exit. That script
// was bash — which made Windows a support problem — and it lived in every
// module folder, so fixing it meant editing every copy. It belongs here: the
// package is versioned, pinned by tag, and Node already knows how to spawn a
// process on three platforms.
//
// Nothing is awaited at startup. The student's first turn is a greeting, and
// uv's first sandbox build can take a minute; the first nb_* call is where
// waiting actually costs something, so that is where the wait happens.
// ---------------------------------------------------------------------------

const MARIMO_BOOT_MS = 180_000;

/** Set when the extension loads; the server lifecycle lives at module level
 *  because process signals do, and it occasionally needs to speak to the
 *  tutor (a page that would not open by itself). */
let piRef: any = null;

let marimoProc: ReturnType<typeof spawn> | null = null;
let marimoStart: Promise<{ url?: string; error?: string }> | null = null;

/** An externally supplied server (the review harness, or an instructor running
 *  marimo by hand) wins: never start a second one.
 *
 *  Read ONCE, at load. marimoUrl() writes our own URL into the same variable
 *  so that nb_run's student code can find it, so anything asking later cannot
 *  tell "the harness gave us a server" from "we started one" — and the restart
 *  path below has to know the difference. */
const EXTERNAL_MARIMO = /^https?:\/\/\S+$/.test(process.env.MARIMO_URL ?? "");
const externalMarimo = () => EXTERNAL_MARIMO;

/** The student's own copy, made on first run so the template stays pristine. */
function bootstrapNotebook(): void {
  try {
    const nb = path.join(process.cwd(), "notebook.py");
    const tpl = path.join(process.cwd(), "notebook.template.py");
    if (!fs.existsSync(nb) && fs.existsSync(tpl)) fs.copyFileSync(tpl, nb);
    // marimo's session snapshot RESTORES pressed buttons, and the cells behind
    // ours have side effects: every send button appends a line to
    // student_signal.txt, which the watcher below reads as the student acting
    // right now. So a session that reopens one of those snapshots replays the
    // last press. A gate run caught the worst version — the ⚖️ appeal replayed
    // one second after boot, and a nervous novice's first turn carried a
    // referee verdict about a case they never filed, on a session where
    // nothing had happened yet. The photo and code send buttons replay the
    // same way: the tutor is told a photo just arrived, and goes looking at
    // the last session's picture. Drop the snapshot before the server starts;
    // a live session re-runs every cell anyway and writes a fresh one.
    const snap = path.join(process.cwd(), "__marimo__", "session", "notebook.py.json");
    if (fs.existsSync(snap)) fs.rmSync(snap, { force: true });
  } catch {
    // startMarimo will fail loudly enough if this mattered
  }
}

function openInBrowser(url: string, onFailure: () => void): void {
  // WSL first on Linux: xdg-open exists there but opens nothing a student sees.
  const attempts: [string, string[]][] =
    process.platform === "darwin"
      ? [["open", [url]]]
      : process.platform === "win32"
        ? [["cmd", ["/c", "start", "", url]]]
        : [
            ["wslview", [url]],
            ["xdg-open", [url]],
            ["explorer.exe", [url]],
          ];
  const tryNext = (i: number) => {
    if (i >= attempts.length) return onFailure();
    const [cmd, args] = attempts[i];
    try {
      const child = spawn(cmd, args, { stdio: "ignore", detached: true });
      let settled = false;
      const next = () => {
        if (settled) return;
        settled = true;
        tryNext(i + 1);
      };
      child.on("error", next);
      // "The command exists" is not "the page opened". On WSL and on a
      // locked-down desktop, xdg-open is present and exits non-zero — and
      // because only the ENOENT path was watched, neither the next opener nor
      // onFailure ever ran. The student was left with a notebook nobody had
      // opened, no kernel, and a tutor that had not been told to say the URL
      // out loud. Give it a moment to fail before believing it.
      child.on("exit", (code) => {
        if (code) next();
        else settled = true;
      });
      const grace = setTimeout(() => (settled = true), 4000);
      (grace as any).unref?.();
      child.unref();
    } catch {
      tryNext(i + 1);
    }
  };
  tryNext(0);
}

function startMarimo(): Promise<{ url?: string; error?: string }> {
  const cwd = process.cwd();
  bootstrapNotebook();
  let log: fs.WriteStream | null = null;
  try {
    fs.mkdirSync(path.join(cwd, "session_artifacts"), { recursive: true });
    // Append, not truncate: a restart that erased the previous server's last
    // words would delete the only record of why it died.
    log = fs.createWriteStream(path.join(cwd, "session_artifacts", "marimo_server.log"), {
      flags: "a",
    });
  } catch {
    // a missing log is survivable; a missing server is not
  }
  return new Promise((resolve) => {
    let settled = false;
    let seen = "";
    let bootTimer: any = null;
    const done = (r: { url?: string; error?: string }) => {
      if (settled) return;
      settled = true;
      // Or the timer holds the event loop open for three minutes after the
      // server is already up, and pi cannot exit when the student says bye.
      if (bootTimer) clearTimeout(bootTimer);
      resolve(r);
    };
    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(
        "uvx",
        ["marimo", "edit", "--sandbox", "--no-token", "--headless", "notebook.py"],
        // Its own process group, so stopMarimo can take down the whole
        // uv -> python -> marimo chain rather than just the wrapper.
        { cwd, detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"] },
      );
    } catch (e: any) {
      return done({ error: `could not start the notebook server: ${e?.message ?? e}` });
    }
    marimoProc = child;
    // The FIRST url marimo prints is not always the one it ends up serving.
    // `--sandbox` makes it re-exec itself inside an isolated uv environment,
    // and the second process binds again — on a different port when the first
    // one is taken, which on a machine running more than one session it often
    // is. Settling on the first match left the tutor talking to a port nothing
    // was listening on, while a perfectly good server ran next door: every
    // nb_* call failed, the student saw "something hiccuped", and the log
    // recorded a URL that answered a curl by hand. Seen twice in one gate run.
    //
    // So a candidate is not a server until it answers. Take the newest URL
    // printed, poll it, and abandon that poll the moment a newer one appears.
    let candidate = "";
    const verify = async (url: string) => {
      for (let i = 0; i < 60 && !settled && candidate === url; i++) {
        try {
          const res = await fetch(`${url}/api/sessions`, {
            signal: AbortSignal.timeout(2_000),
          });
          if (res.ok) return done({ url });
        } catch {
          // not up yet, or not this port at all
        }
        await new Promise((r) => setTimeout(r, 500));
      }
    };
    const scan = (chunk: Buffer) => {
      log?.write(chunk);
      seen += chunk.toString();
      const all = seen.match(/http:\/\/[A-Za-z0-9.\-]+:\d+/g) ?? [];
      const latest = all[all.length - 1];
      if (latest && latest !== candidate) {
        candidate = latest;
        void verify(latest);
      }
    };
    child.stdout?.on("data", scan);
    child.stderr?.on("data", scan);
    child.on("error", (e: any) =>
      done({
        error:
          e?.code === "ENOENT"
            ? "uv is not installed, so the notebook cannot start (install it: https://docs.astral.sh/uv/)"
            : `could not start the notebook server: ${e?.message ?? e}`,
      }),
    );
    child.on("exit", (code) =>
      done({
        error: `the notebook server stopped (exit ${code}) — session_artifacts/marimo_server.log has why`,
      }),
    );
    bootTimer = setTimeout(
      () => done({ error: "the notebook server did not come up in time" }),
      MARIMO_BOOT_MS,
    );
  });
}

/** Start the server once, remember the result, open the student's page. */
function marimoUrl(): Promise<{ url?: string; error?: string }> {
  if (externalMarimo()) return Promise.resolve({ url: marimoBase() });
  if (!marimoStart) {
    marimoStart = startMarimo().then((r) => {
      if (!r.url) return r;
      // Everything downstream reads the env var, including a nb_run the
      // student's own code might make.
      process.env.MARIMO_URL = r.url;
      openInBrowser(`${r.url}/?view-as=present`, () => {
        // No browser opener on this machine — the tutor has to say it out loud,
        // because a notebook nobody has open is a kernel that never wakes.
        try {
          piRef?.sendMessage(
            {
              customType: "notebook-url",
              content:
                `NOTE (invisible to the student): their notebook page did not open by itself. ` +
                `Tell them in ONE sentence to open ${r.url}/?view-as=present in their browser, ` +
                `then carry on.`,
              display: false,
            },
            { deliverAs: "nextTurn" },
          );
        } catch {
          /* best effort */
        }
      });
      return r;
    });
  }
  return marimoStart;
}

/**
 * Bring a server we started back after it died, and reopen the student's page.
 *
 * A lesson runs 60-90 minutes and outlives a lot: the laptop sleeps, the tab
 * gets closed, marimo is OOM-killed. Before this existed any one of those
 * ended the notebook for the rest of the session — the tutor said "the
 * whiteboard needs a restart" (it is told to), the student had no way to
 * restart it, and every remaining checkpoint went unbuilt while the graded
 * artifact stopped growing. Killing marimo mid-checkpoint in a live run left
 * exactly that: a session that talked its way to the end with nothing on the
 * page.
 *
 * Never touches a server someone else started — the review harness pins one
 * on purpose, and an instructor running marimo by hand would not thank us.
 * Rate-limited, so a genuinely unstartable server (no uv, a busy port) costs
 * one boot attempt per minute rather than one per tool call.
 */
const RELAUNCH_COOLDOWN_MS = 60_000;
let lastRelaunch = 0;
let lastReopen = 0;

/**
 * Put the student's notebook tab back. A marimo kernel exists only while a
 * browser client is attached, so a closed tab is indistinguishable, from
 * here, from a dead server — except that it is far commoner and far cheaper
 * to fix. Works for a harness-supplied server too: reopening someone else's
 * page costs them nothing.
 */
function reopenPage(): boolean {
  if (Date.now() - lastReopen < RELAUNCH_COOLDOWN_MS) return false;
  lastReopen = Date.now();
  try {
    openInBrowser(`${marimoBase()}/?view-as=present`, () => {});
    return true;
  } catch {
    return false;
  }
}

async function relaunchMarimo(): Promise<boolean> {
  if (externalMarimo()) return false;
  if (Date.now() - lastRelaunch < RELAUNCH_COOLDOWN_MS) return false;
  lastRelaunch = Date.now();
  stopMarimo(); // clears the cached start promise, and reaps a half-dead chain
  try {
    const r = await marimoUrl();
    return !!r.url;
  } catch {
    return false;
  }
}

function stopMarimo(): void {
  const child = marimoProc;
  marimoProc = null;
  marimoStart = null;
  if (!child?.pid) return;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      process.kill(-child.pid, "SIGTERM"); // the group, not just uvx
    }
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {
      /* it is already gone */
    }
  }
}

// session_shutdown is the normal path; these catch the rest (Ctrl-C, a crash),
// because a marimo left running holds port 2718 against the next session.
for (const sig of ["exit", "SIGINT", "SIGTERM", "SIGHUP"] as const) {
  process.on(sig, () => stopMarimo());
}

/** Guard against a garbage env value (a broken grep once exported
 *  "Binary file ... matches" as the URL and every call failed). */
function marimoBase(): string {
  const envUrl = process.env.MARIMO_URL ?? "";
  const url = /^https?:\/\/\S+$/.test(envUrl) ? envUrl : "http://127.0.0.1:2718";
  return url.replace(/\/+$/, "");
}

function marimoHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = process.env.MARIMO_TOKEN;
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra };
}

/** What the tutor is told when the notebook cannot be reached at all. */
const NO_NOTEBOOK =
  "The notebook is not reachable, so nothing can be built or read there right now. " +
  "Tell the student, in one warm sentence, that the notebook needs restarting — then " +
  "keep teaching in the terminal.";

/**
 * Which kernel session to run in. Resolved per call, never cached: marimo
 * renames a session when the browser reconnects, and a stale id fails in a way
 * that looks like broken code rather than a broken connection.
 */
type SessionLookup = {
  id?: string;
  error?: string;
  /** Why it failed, so the caller can pick the right repair: a server that is
   *  gone needs restarting, a page the student closed only needs reopening —
   *  and restarting a healthy server to fix a closed tab costs the student a
   *  minute and a half of silence for nothing. */
  reason?: "unreachable" | "no-page" | "ambiguous";
};

async function resolveSession(signal: AbortSignal): Promise<SessionLookup> {
  let sessions: Record<string, { path?: string; filename?: string }> = {};
  let ids: string[] = [];
  // A marimo kernel exists only while a browser client is attached. The page is
  // opened for the student the moment the server reports its URL, but a cold
  // laptop can take a while to render it — and the first nb_* call can easily
  // arrive first. Waiting here is invisible; failing here costs the tutor a
  // checkpoint and the student an apology for something that was about to work.
  const deadline = Date.now() + SESSION_WAIT_MS;
  // If nothing has attached after a few seconds, the likeliest reason is not
  // a slow laptop but a tab that was closed — students close tabs. Put it
  // back while we are still waiting, rather than spending the full ninety
  // seconds first and only then noticing. Rate-limited inside reopenPage, so
  // a machine with no browser opener at all does not get spammed.
  const nudgeAt = Date.now() + 8_000;
  let nudged = false;
  for (;;) {
    try {
      const res = await fetch(`${marimoBase()}/api/sessions`, {
        headers: marimoHeaders(),
        signal,
      });
      if (!res.ok)
        return { error: `${NO_NOTEBOOK} (server said ${res.status})`, reason: "unreachable" };
      sessions = (await res.json()) as typeof sessions;
    } catch {
      return { error: NO_NOTEBOOK, reason: "unreachable" };
    }
    ids = Object.keys(sessions ?? {});
    if (ids.length > 0) break;
    if (Date.now() >= deadline || signal.aborted) {
      return {
        reason: "no-page",
        error:
          "The notebook page is not open, so its kernel is asleep and nothing can be built " +
          "there. Ask the student to open the notebook tab (it may have been closed) — and " +
          "keep teaching in the terminal meanwhile.",
      };
    }
    if (!nudged && Date.now() >= nudgeAt) {
      nudged = true;
      reopenPage();
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  if (ids.length === 1) return { id: ids[0] };
  // More than one notebook on this server: pick ours by path. Exact first —
  // the basename rule matches any notebook.py anywhere, which is a fallback
  // for a marimo that reports paths differently, not a way to choose.
  const want = path.join(process.cwd(), "notebook.py");
  const exact = ids.filter((id) => {
    const s = sessions[id] ?? {};
    return s.path === want || s.filename === want;
  });
  const mine = exact.length
    ? exact
    : ids.filter((id) => {
        const s = sessions[id] ?? {};
        return path.basename(s.path ?? s.filename ?? "") === "notebook.py";
      });
  // Ambiguity used to be fatal, and fatal here means fatal for the session:
  // the error carried NO_NOTEBOOK, which tells the tutor to announce that the
  // whiteboard needs restarting and finish in the terminal — over a server
  // that is running, healthy, and holding the student's own notebook. Guessing
  // wrong costs one cell in the wrong page; refusing costs the artifact. So
  // take the first match and keep going.
  if (mine.length >= 1) return { id: mine[0] };
  return {
    reason: "ambiguous",
    error: `No open notebook matches this folder (${ids.length} on the server). ${NO_NOTEBOOK}`,
  };
}

/**
 * Run Python in the notebook's scratchpad and return everything it printed.
 *
 * `failed` is true when the kernel reported an error, when the stream ended
 * without a `done` event (the code never ran), or when the server could not be
 * reached — every caller treats those the same way: surface the RECOVERY line
 * to the tutor, never to the student.
 */
async function runKernel(
  code: string,
  signal?: AbortSignal,
): Promise<{ out: string; failed: boolean; reason?: string }> {
  const timeout = AbortSignal.timeout(KERNEL_TIMEOUT_MS);
  const abort = signal ? AbortSignal.any([signal, timeout]) : timeout;

  // The server was started at load; this is where the first call waits for it.
  const server = await marimoUrl();
  if (!server.url) return { out: `${NO_NOTEBOOK} (${server.error})`, failed: true };

  let session = await resolveSession(abort);
  // A closed tab is already handled inside resolveSession, which puts the page
  // back mid-wait. What is left here is the server itself being gone: start it
  // again, once, and look for the kernel a second time.
  if (!session.id && !abort.aborted && session.reason === "unreachable") {
    if (await relaunchMarimo()) session = await resolveSession(abort);
  }
  if (!session.id)
    return { out: session.error ?? NO_NOTEBOOK, failed: true, reason: session.reason };

  let res: Response;
  try {
    res = await fetch(`${marimoBase()}/api/kernel/execute`, {
      method: "POST",
      headers: marimoHeaders({
        "Content-Type": "application/json",
        "Marimo-Session-Id": session.id,
      }),
      body: JSON.stringify({ code }),
      signal: abort,
    });
  } catch {
    return { out: NO_NOTEBOOK, failed: true };
  }
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    return { out: `${NO_NOTEBOOK} (server said ${res.status}) ${detail}`.trim(), failed: true };
  }

  // SSE: "event: <name>" then "data: <json>", records separated by a blank
  // line. stdout/stderr arrive as {"data": "..."} and concatenate without
  // separators — they are a byte stream, not lines.
  let stdout = "";
  let stderr = "";
  let failed = false;
  let done = false;
  let unparsed = "";
  let event = "";
  let buffer = "";

  const handle = (line: string) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      return;
    }
    if (line === "") return;
    if (!line.startsWith("data:")) {
      // Not SSE at all — an error body rather than a stream.
      unparsed += line + "\n";
      return;
    }
    const raw = line.slice(5).trim();
    let payload: any;
    try {
      payload = JSON.parse(raw);
    } catch {
      unparsed += raw + "\n";
      return;
    }
    switch (event) {
      case "stdout":
        stdout += payload?.data ?? "";
        break;
      case "stderr":
        stderr += payload?.data ?? "";
        break;
      case "done":
        // Carries the success bit and the last expression's output; errors
        // already arrived as stderr.
        if (payload?.success === false) failed = true;
        else if (payload?.output?.data) stdout += payload.output.data + "\n";
        done = true;
        break;
    }
  };

  try {
    const decoder = new TextDecoder();
    for await (const chunk of res.body as any) {
      buffer += decoder.decode(chunk as Uint8Array, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        handle(buffer.slice(0, nl).replace(/\r$/, "")); // SSE permits CRLF
        buffer = buffer.slice(nl + 1);
      }
    }
    if (buffer) handle(buffer.replace(/\r$/, ""));
  } catch {
    return {
      out: [stdout, stderr, "The notebook connection dropped mid-run."].filter(Boolean).join("\n"),
      failed: true,
    };
  }

  if (!done) {
    failed = true;
    stderr +=
      "\nExecution did not complete: the server ended the stream without a result." +
      (unparsed ? `\n${unparsed.trim()}` : "");
  }
  return { out: [stdout, stderr].filter(Boolean).join("\n").trim(), failed };
}

/**
 * Run all notebook cells if the kernel hasn't executed them yet. Must be a
 * SEPARATE kernel call from any cell create/edit: queued runs inside one
 * code-mode context do not reliably execute before a newly created cell
 * (observed in production: new cell ran first and hit NameError on `mo`).
 */
async function ensureWarm(signal?: AbortSignal): Promise<{ out: string; failed: boolean } | null> {
  const probe = await runKernel(`print("OK" if "mo" in globals() else "COLD")`, signal);
  if (probe.failed) return probe;
  if (probe.out.includes("COLD")) {
    const boot = await runKernel(BOOTSTRAP, signal);
    if (boot.failed) return boot;
  }
  return null;
}

/**
 * The tutor model (glm-5.2 cloud) is text-only — nb_view_image delegates
 * "seeing" to a separate vision-capable model. Resolution order:
 *   1. TUTOR_VISION_MODEL env ("provider/model-id", as pi knows it)
 *   2. an image-capable model on the tutor's own provider (same billing)
 *   3. any zero-cost image-capable model (local servers are safe to auto-pick)
 * None found -> the tool tells the tutor to ask for a verbal description.
 */
function resolveVisionModel(ctx: any): any | null {
  const reg = ctx?.modelRegistry;
  if (!reg) return null;
  const all: any[] = reg.getAvailable?.() ?? [];
  const pinned = (process.env.TUTOR_VISION_MODEL ?? "").trim();
  if (pinned.includes("/")) {
    const i = pinned.indexOf("/");
    const m = reg.find?.(pinned.slice(0, i), pinned.slice(i + 1));
    if (m) return m;
    // Router model ids contain slashes themselves ("openrouter/minimax/minimax-m3");
    // accept the full provider/id form or the bare router slug, case-insensitively.
    const want = pinned.toLowerCase();
    const byId = all.find(
      (c) =>
        `${c.provider}/${c.id}`.toLowerCase() === want || String(c.id).toLowerCase() === want,
    );
    if (byId) return byId;
  }
  const canSee = (m: any) => Array.isArray(m?.input) && m.input.includes("image");
  const sameProvider = all.find((m) => canSee(m) && m.provider === ctx?.model?.provider);
  if (sameProvider) return sameProvider;
  return (
    all.find((m) => canSee(m) && m?.cost && m.cost.input === 0 && m.cost.output === 0) ?? null
  );
}

async function describeImage(
  ctx: any,
  b64jpeg: string,
  task: string,
  question: string,
): Promise<{ text: string; model?: string; failed: boolean }> {
  const noVisionAdvice =
    "Ask the student to describe their drawing in words instead (e.g. which dots " +
    "they connected and why), then judge their words — that is a perfectly valid pass. " +
    "Say NOTHING to them about the picture not being readable: a live run announced " +
    "\"the page viewer's being stubborn today\", which is your plumbing and not their " +
    "problem. Just ask them to talk you through the page.";
  const model = resolveVisionModel(ctx);
  if (!model) {
    return {
      failed: true,
      text:
        `NO VISION MODEL: you are text-only and no vision-capable model is configured ` +
        `(instructor: set TUTOR_VISION_MODEL=provider/model-id). ${noVisionAdvice}`,
    };
  }
  try {
    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
    if (!auth?.ok) throw new Error(auth?.error ?? "no credentials for vision model");
    // A bare "describe the image" fails on messy hand drawings (a chord two
    // steps apart was reported as "already neighbors" in production). The
    // model needs the TASK to know what to look for, and a forced
    // shape-by-shape / line-by-line trace before answering.
    const prompt =
      "You are the eyes of a text-only tutor looking at a student's photo.\n" +
      `What the student was asked to do: ${task}\n` +
      "Work carefully:\n" +
      "1. Count the main shapes (dots, boxes...) and name each by its position — " +
      "clock positions work well for anything arranged in a circle.\n" +
      "2. Trace EVERY line one at a time; for each, name the two things it connects.\n" +
      "3. Then answer the tutor's question using those position names.\n" +
      "If you are unsure about anything, say so explicitly instead of guessing.\n" +
      "Describe only — never grade or judge. Under 150 words.\n" +
      `Tutor's question: ${question}`;
    const response = await complete(
      model,
      {
        messages: [
          {
            role: "user" as const,
            content: [
              { type: "text" as const, text: prompt },
              { type: "image" as const, data: b64jpeg, mimeType: "image/jpeg" },
            ],
            timestamp: Date.now(),
          },
        ],
      },
      {
        apiKey: auth.apiKey,
        headers: auth.headers,
        env: auth.env,
        // Vision models that can think should: tracing lines in a wobbly
        // hand drawing is exactly what fails without it.
        ...(model.reasoning ? { reasoningEffort: "medium" as const } : {}),
        cacheRetention: "none",
        sessionId: uuidv7(),
      },
    );
    const text = (response?.content ?? [])
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("\n")
      .trim();
    if (!text) throw new Error("vision model returned no text");
    return { failed: false, model: `${model.provider}/${model.id}`, text };
  } catch (e: any) {
    return {
      failed: true,
      text:
        `VISION FAILED (${e?.message ?? e}): you cannot see the image this time. ` +
        `Do not retry more than once and do not debug. ${noVisionAdvice}`,
    };
  }
}

// ── The referee (the student's ⚖️ "Tutor gets stuck" appeal) ────────────────
// The notebook carries a persistent "Tutor gets stuck — call the referee"
// box. Pressing it appends "tutor_stuck" to student_signal.txt (their case
// text goes to session_artifacts/appeal.txt); the watcher hands the whole
// situation to a STRONGER model, whose ruling is delivered to the tutor as a
// binding "REFEREE VERDICT" message. The verdict can order what the tutor
// cannot grant alone — count the work already shown, redo on fresh data,
// skip ahead, end the chapter — so rulings that close things also arm a
// short-lived waiver that the photo/build/chapter gates honor.

/** ctx from the most recent hook/tool call — the referee runs outside any
 * tool, but needs ctx.modelRegistry for model lookup and credentials. */
let lastCtx: any = null;

let refereeWaiver: { ruling: string; expires: number } | null = null;
const refereeWaiverActive = (): boolean =>
  !!refereeWaiver && refereeWaiver.expires > Date.now();

let appealInFlight = false;

const REFEREE_RULINGS = [
  "keep_going",
  "hear_them_out",
  "redo_fresh",
  "accept_and_close",
  "skip_ahead",
  "next_chapter",
];

/** Rulings that close or skip things need the gates to stand aside. */
const WAIVER_RULINGS = ["accept_and_close", "skip_ahead", "next_chapter"];

function resolveRefereeModel(ctx: any): any | null {
  const reg = ctx?.modelRegistry;
  if (!reg) return null;
  // Default to a "referee" model on the tutor's OWN provider — a course
  // gateway publishes tutor / vision / referee as three aliases, and the
  // student has credentials for exactly that provider and no other. The
  // launcher used to export this; nothing exports anything now.
  const sameProvider = ctx?.model?.provider
    ? (reg.find?.(ctx.model.provider, "referee") ?? null)
    : null;
  const pinned = (
    process.env.TUTOR_REFEREE_MODEL ??
    (sameProvider ? `${sameProvider.provider}/${sameProvider.id}` : "openrouter/z-ai/glm-5.2")
  ).trim();
  if (!pinned.includes("/")) return null;
  const i = pinned.indexOf("/");
  const m = reg.find?.(pinned.slice(0, i), pinned.slice(i + 1));
  if (m) return m;
  // Router model ids contain slashes themselves ("openrouter/z-ai/glm-5.2");
  // accept the full provider/id form or the bare router slug.
  const want = pinned.toLowerCase();
  const all: any[] = reg.getAvailable?.() ?? [];
  return (
    all.find(
      (c) =>
        `${c.provider}/${c.id}`.toLowerCase() === want || String(c.id).toLowerCase() === want,
    ) ?? null
  );
}

/** The recent conversation, labeled and clipped, for the referee's eyes. */
function transcriptTail(ctx: any, maxMsgs = 40, maxChars = 7000): string {
  const entries: any[] = ctx?.sessionManager?.getBranch?.() ?? [];
  const rows: string[] = [];
  for (const e of entries) {
    if (e?.type !== "message") continue;
    const role = e?.message?.role;
    if (role !== "user" && role !== "assistant") continue;
    const c = e.message.content;
    const text = (
      typeof c === "string"
        ? c
        : Array.isArray(c)
          ? c
              .filter((p: any) => p?.type === "text" && typeof p.text === "string")
              .map((p: any) => p.text)
              .join("\n")
          : ""
    ).trim();
    if (!text) continue;
    if (role === "user" && INJECTED_PREFIX.test(text)) continue;
    rows.push(
      `${role === "user" ? "STUDENT" : "TUTOR"}: ${text.length > 500 ? text.slice(0, 500) + "…" : text}`,
    );
  }
  let tail = rows.slice(-maxMsgs);
  while (tail.join("\n").length > maxChars && tail.length > 4) tail = tail.slice(2);
  return tail.join("\n") || "(no conversation captured)";
}

function currentChapterScriptRaw(): string {
  try {
    const dir = path.join(process.cwd(), "lesson");
    const idx = JSON.parse(fs.readFileSync(path.join(dir, "index.json"), "utf-8"));
    const cur = currentChapterId() ?? idx.chapters?.[0]?.id;
    const ch = (idx.chapters ?? []).find((c: any) => c.id === cur) ?? idx.chapters?.[0];
    if (!ch?.file) return "(no chapter script found)";
    return fs.readFileSync(path.join(dir, ch.file), "utf-8").slice(0, 16000);
  } catch {
    return "(no chapter script found)";
  }
}

async function handleAppeal(pi: any): Promise<void> {
  if (appealInFlight) return;
  appealInFlight = true;
  try {
    let caseText = "(no details given)";
    try {
      caseText =
        fs
          .readFileSync(path.join(process.cwd(), "session_artifacts", "appeal.txt"), "utf-8")
          .trim() || caseText;
    } catch {
      /* the button works even if the case file does not */
    }
    const ctx = lastCtx;
    const model = ctx ? resolveRefereeModel(ctx) : null;
    let verdict: {
      ruling: string;
      reason: string;
      directive: string;
      to_student: string;
    } | null = null;
    let modelName = "";
    if (model) {
      try {
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
        if (!auth?.ok) throw new Error(auth?.error ?? "no credentials for referee model");
        const prompt =
          `You are the REFEREE for a live one-on-one tutoring session (a Socratic AI ` +
          `tutor teaching a university network-science module to a possibly ` +
          `non-programming student). The student just pressed a "Tutor gets stuck" ` +
          `button: they are appealing over the tutor's head, and your ruling is ` +
          `binding on the tutor.\n\n` +
          `The tutor's standing rules, for context: it never states the answer to an ` +
          `open checkpoint; hints are unlimited and never penalized; predictions are ` +
          `never wrong; typed work substitutes for photographed pen-and-paper work ` +
          `when photographing is a hardship; honest reasoning deserves credit; extra ` +
          `practice is never failure. Rules serve the student — where a rule's letter ` +
          `and the student's legitimate progress conflict, side with the student.\n\n` +
          // The six rulings below all decide a CHECKPOINT, and an appeal is not
          // always about one. A student appealed because the tutor would not
          // hand their work in to GitHub; the referee had no ruling for that,
          // ruled on the nearest checkpoint instead, and the tutor accepted the
          // ruling and moved on — the request itself was never granted. The
          // directive is the channel for anything the rulings do not name.
          `WHAT THE TUTOR CAN ACTUALLY DO, so you do not rule around it: it builds and ` +
          `edits notebook cells, reads what the student typed into a widget, looks at an ` +
          `uploaded photo, logs checkpoints and detours, and HANDS THE WORK IN — it has ` +
          `an nb_submit tool that commits and pushes their notebook and session log to ` +
          `their GitHub repository, and it may use it at any point, not only at the end. ` +
          `If the appeal is a REQUEST the tutor refused or dropped rather than a dispute ` +
          `about a checkpoint, rule "keep_going" and make the directive the action: name ` +
          `the tool and say to do it now, before anything else.\n\n` +
          `THE STUDENT'S CASE (verbatim):\n${caseText}\n\n` +
          `PROGRESS SO FAR (graded log):\n${progressBrief(readSessionLog())}\n\n` +
          `CURRENT CHAPTER SCRIPT (the tutor's curriculum for right now):\n` +
          `${currentChapterScriptRaw()}\n\n` +
          `RECENT CONVERSATION (oldest first; STUDENT lines are the student's own ` +
          `words):\n${transcriptTail(ctx)}\n\n` +
          `Decide the fairest way forward. Choose EXACTLY ONE ruling:\n` +
          `- "keep_going": the tutor is handling it right; name ONE concrete ` +
          `adjustment it should make.\n` +
          `- "hear_them_out": the student deserves a proper hearing on this ` +
          `checkpoint; tell the tutor what to ask and what standard to accept.\n` +
          `- "redo_fresh": scrap the current attempt; restart the SAME kind of ` +
          `problem on fresh data.\n` +
          `- "accept_and_close": the work already shown meets the bar; the current ` +
          `checkpoint closes now, judged on what actually happened.\n` +
          `- "skip_ahead": close the current checkpoint as unresolved and move to ` +
          `the next one; no grade penalty.\n` +
          `- "next_chapter": end this chapter now and move on to the next.\n\n` +
          `Answer with ONLY a JSON object, no prose around it:\n` +
          `{"ruling": "<one of the six>", "reason": "1-2 plain sentences", ` +
          `"directive": "2-4 concrete sentences the tutor must now follow, naming ` +
          `the checkpoint(s) to act on", "to_student": "1-2 warm sentences addressed ` +
          `directly to the student"}`;
        const response = await complete(
          model,
          {
            messages: [
              {
                role: "user" as const,
                content: [{ type: "text" as const, text: prompt }],
                timestamp: Date.now(),
              },
            ],
          },
          {
            apiKey: auth.apiKey,
            headers: auth.headers,
            env: auth.env,
            ...(model.reasoning ? { reasoningEffort: "medium" as const } : {}),
            cacheRetention: "none",
            sessionId: uuidv7(),
          },
        );
        const text = (response?.content ?? [])
          .filter((c: any) => c.type === "text")
          .map((c: any) => c.text)
          .join("\n")
          .trim();
        const m = text.match(/\{[\s\S]*\}/);
        let parsed: any = null;
        try {
          parsed = m ? JSON.parse(m[0]) : null;
        } catch {
          parsed = null;
        }
        if (parsed && REFEREE_RULINGS.includes(parsed.ruling)) {
          verdict = {
            ruling: String(parsed.ruling),
            reason: String(parsed.reason ?? ""),
            directive: String(parsed.directive ?? ""),
            to_student: String(parsed.to_student ?? ""),
          };
          modelName = `${model.provider}/${model.id}`;
        } else if (text) {
          // A referee that replied in prose still ruled — treat the whole
          // reply as the directive rather than dropping the appeal.
          verdict = {
            ruling: "hear_them_out",
            reason: "(referee replied in prose)",
            directive: text.slice(0, 1200),
            to_student: "",
          };
          modelName = `${model.provider}/${model.id}`;
        }
      } catch {
        /* fall through to the unreachable-referee path */
      }
    }
    // The artifact records every appeal, ruled or not, before the tutor
    // hears about it — misbehavior after the verdict cannot erase it.
    appendLog({
      type: "appeal",
      student_case: caseText,
      ruling: verdict?.ruling ?? "unresolved_no_referee",
      reason: verdict?.reason ?? "",
      referee: modelName || "(unreachable)",
    });
    if (!verdict || WAIVER_RULINGS.includes(verdict.ruling)) {
      refereeWaiver = {
        ruling: verdict?.ruling ?? "unresolved_no_referee",
        expires: Date.now() + 20 * 60_000,
      };
    }
    const content = verdict
      ? `REFEREE VERDICT (binding). The student pressed the ⚖️ "Tutor gets stuck" ` +
        `button, and a stronger referee model reviewed the whole situation. Its ` +
        `ruling overrides your script and your own judgment wherever they ` +
        `conflict.\n` +
        `Their case, in their words: "${caseText}"\n` +
        `Ruling: ${verdict.ruling} — ${verdict.reason}\n` +
        (verdict.to_student
          ? `For the student (deliver this warmly, in your own words, as the ` +
            `referee's decision): ${verdict.to_student}\n`
          : "") +
        `What you do now, exactly: ${verdict.directive}\n` +
        // Accepting a ruling is not carrying it out. A student appealed
        // because the tutor would not hand their work in; the tutor accepted
        // the verdict warmly, said nothing more about it, and moved on to the
        // next chapter with the request never granted. Whatever the ruling
        // names, the thing the student ASKED for is still owed.
        `And whatever the ruling says, THEIR ORIGINAL REQUEST IS STILL OWED. Agreeing ` +
        `with the referee is not doing the thing: if they asked for their work to be ` +
        `handed in, call nb_submit; if they asked for another problem, set one. Do it in ` +
        `this turn, before you move on, and never advance a chapter with their request ` +
        `still unanswered.\n` +
        `Do not argue with or re-litigate the ruling, and never hold the appeal ` +
        `against the student — appealing is participation. The appeal itself is ` +
        `already in the log; close any affected checkpoint honestly (its row is ` +
        `stamped for the graders). If a gate refused you before, it will let this ` +
        `ruling through now.`
      : `REFEREE VERDICT (the referee could not be reached — you resolve this ` +
        `appeal yourself). The student pressed the ⚖️ "Tutor gets stuck" button ` +
        `with this case, in their words: "${caseText}"\n` +
        `Resolve it generously, in their favor where reasonable: react to their ` +
        `case, then offer plainly the three ways forward — count the honest work ` +
        `they have already shown, redo the problem on fresh data, or simply move ` +
        `on — and do what they choose. The appeal is already in the log; log ` +
        `whatever closes as it actually happened (notes may say "resolved without ` +
        `referee").`;
    pi.sendMessage(
      { customType: "referee-verdict", content, display: false },
      { deliverAs: "followUp", triggerTurn: true },
    );
  } catch {
    /* an appeal must never break the session */
  } finally {
    appealInFlight = false;
  }
}

function toResult({ out, failed, reason }: { out: string; failed: boolean; reason?: string }) {
  let text = failed ? `NOTEBOOK ERROR:\n${out || "(no output)"}` : out || "(ok)";
  if (failed) {
    // Tell the model exactly what to do — otherwise it starts "debugging"
    // with skills, shell, and log files in front of the student.
    //
    // This used to branch on out.includes("No active sessions") — a string
    // that appears nowhere else in this repo and that marimo never returns.
    // So the one case with a real, cheap fix (their tab is closed; ask them to
    // open it) always fell through to "tell them the whiteboard is
    // unavailable and continue in terminal-only mode", which ends the notebook
    // for the rest of the session over a browser tab. The reason now travels
    // as a field instead of being guessed from prose.
    text +=
      reason === "no-page"
        ? `\nRECOVERY: the notebook server is fine — the student's notebook TAB is closed, ` +
          `so the page has no kernel. Ask them, in one warm sentence, to open ` +
          `${marimoBase()}/?view-as=present (it may already be reopening by itself), wait ` +
          `for their reply, then retry this call. Do NOT switch to terminal-only mode for ` +
          `this and do NOT tell them the whiteboard is broken.`
        : `\nRECOVERY: retry this call ONCE. If it fails again, tell the student the ` +
          `whiteboard is unavailable and continue in terminal-only mode (AGENTS.md) — ` +
          `do NOT investigate with skills, shell, or log files.`;
  }
  return {
    content: [{ type: "text" as const, text }],
    details: { failed },
  };
}

// ── Is the tutor's own model actually reachable? ────────────────────────────
// The course endpoint moved once, and every machine still pointing at the old
// host got, in full, this: "Error: Connection error." — four times, then
// "Retry failed after 3 attempts". No URL, no status, no mention of a key, no
// suggestion. On a student who has never used a terminal that is the end of
// the assignment, and the instructor hears "it doesn't work".
//
// So the session opens by asking the provider whether it is there. It costs
// one HTTP call against an endpoint that must work anyway, it happens before
// the student has typed anything, and it names the three things that are ever
// wrong: the address, the key, or the network. Silent when all is well.
async function preflightProvider(ctx: any): Promise<string | null> {
  const model = ctx?.model;
  const provider = model?.provider;
  if (!provider) return null;
  let auth: any;
  try {
    // Same call the vision path uses: {ok, apiKey, headers, baseUrl}.
    auth = await ctx.modelRegistry?.getApiKeyAndHeaders?.(model);
  } catch {
    return null;
  }
  if (!auth?.ok) return null;
  const baseUrl = String(
    auth.baseUrl ?? ctx.modelRegistry?.getProvider?.(provider)?.baseUrl ?? "",
  ).replace(/\/+$/, "");
  // A built-in provider with no configured baseUrl of its own is not what this
  // is for — those speak their vendor's endpoint and diagnose themselves.
  if (!/^https?:\/\//.test(baseUrl)) return null;

  const fix =
    `What to do: run  node setup-pi.mjs  in this folder — it rewrites the course ` +
    `settings and tells you which part is wrong. If it still fails, send your ` +
    `instructor the line above.`;
  try {
    const res = await fetch(`${baseUrl}/models`, {
      headers: {
        ...(auth.headers ?? {}),
        ...(auth.apiKey ? { Authorization: `Bearer ${auth.apiKey}` } : {}),
      },
      signal: AbortSignal.timeout(15_000),
    });
    if (res.ok) return null;
    if (res.status === 401 || res.status === 403) {
      return (
        `Your course key was refused by ${baseUrl} (HTTP ${res.status}).\n` +
        `Check NETSCI_API_KEY in this terminal — a missing character or a stray ` +
        `space is the usual cause, and a key set in another window does not ` +
        `count.\n${fix}`
      );
    }
    if (res.status === 429) {
      return (
        `The course server says you are over your allowance for now (HTTP 429).\n` +
        `See what is left at ${baseUrl}/usage, and tell your instructor if that ` +
        `looks wrong.`
      );
    }
    return `The course server answered ${baseUrl} with HTTP ${res.status}.\n${fix}`;
  } catch (e: any) {
    const why =
      e?.name === "TimeoutError" ? "it did not answer in 15 seconds" : (e?.message ?? String(e));
    return (
      `Could not reach the course server at ${baseUrl} — ${why}.\n` +
      `If you are online, the address in your settings may be out of date.\n${fix}`
    );
  }
}

// ── Pickers the student can type into ──────────────────────────────────────
// pi's ctx.ui.select swallows every printable key. A student who answers by
// typing while a picker is open sees NOTHING appear on screen — no echo, no
// warning — and their Enter then selects whatever row the cursor happens to
// be on. Both halves of that were reproduced in one live run: the sentence
// was gone and the choice was not theirs. The tutor's own contract warns the
// TUTOR about this ("a picker takes the keyboard"), but the student, who is
// the one holding the keyboard, was never protected at all.
//
// So every picker this toolkit opens carries a typed-answer row, and its
// title says how to drive it. The row cannot rescue keys pressed before the
// student notices — nothing can, from outside pi — but it does mean there is
// always somewhere for their words to go, and that a beginner who types out
// of habit finds the way through in the list they are already looking at.
const TYPE_IT = "✎ Let me type something instead";

/**
 * True while a picker is on screen, read by the trivia timer.
 *
 * A tip is for dead air — the long wait on a model call or a vision read —
 * and the seconds around a picker are the opposite of dead air. The tutor has
 * just delivered the reveal, and every line printed under it pushes it further
 * up a terminal the student may not think to scroll. Live, in order: the
 * reveal, a tool status line, a tip, a separator, the picker. On a short
 * window the payoff is off the top and the student answers "Where to next?"
 * having never read what their answer bought them.
 */
let quietForPicker = false;

/**
 * The last thing the tutor SAID, short enough to sit on top of a picker.
 *
 * The reveal is the payoff for the answer the student just gave, and the
 * picker that follows it is titled "Where to next?" and nothing else — so
 * when the reveal has scrolled off the top, the student is choosing with no
 * idea what they were shown. Carrying the last line into the dialog costs a
 * line of repetition when it is still visible, and saves the whole beat when
 * it is not.
 */
function lastTutorLine(ctx: any, maxChars = 150): string {
  const entries: any[] = ctx?.sessionManager?.getBranch?.() ?? [];
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i];
    if (e?.type !== "message" || e?.message?.role !== "assistant") continue;
    const c = e.message.content;
    const text = (
      typeof c === "string"
        ? c
        : Array.isArray(c)
          ? c
              .filter((p: any) => p?.type === "text" && typeof p.text === "string")
              .map((p: any) => p.text)
              .join("\n")
          : ""
    ).trim();
    if (!text) continue;
    // The last SENTENCE, not the last line. A reveal is two sentences on one
    // line as often as not, and cutting that at a character count ends the
    // dialog mid-clause — "…at most two odd nodes for a trail, and none at…",
    // which is worse than showing nothing. The final sentence is also the one
    // the beat lands on: "Königsberg has four odd nodes, so it has neither."
    const flat = text.replace(/\s+/g, " ").trim();
    const parts = flat.split(/(?<=[.!?])\s+/).filter(Boolean);
    // The last two sentences when they fit, not just the last one. A reveal
    // is two sentences by contract, and the last of them is often the optional
    // extra beat — "And the citizens had asked for the circuit" — while the
    // punchline is the one before it. Taking both, when both fit, is the whole
    // beat rather than its tail.
    let line = parts[parts.length - 1] ?? flat;
    if (parts.length > 1) {
      const pair = `${parts[parts.length - 2]} ${line}`;
      if (pair.length <= maxChars) line = pair;
    }
    return line.length > maxChars ? line.slice(0, maxChars - 1).trimEnd() + "…" : line;
  }
  return "";
}

async function askStudent(
  ctx: any,
  title: string,
  options: string[],
): Promise<{ choice: string | null; typed: string; asked: boolean }> {
  if (!ctx?.ui?.select) return { choice: null, typed: "", asked: false };
  quietForPicker = true;
  // Silencing the timer is not enough: a tip set moments earlier is still on
  // screen, and it sits between the reveal and the picker for as long as the
  // student takes to choose. Clear it as the dialog opens.
  try {
    ctx?.ui?.setWorkingMessage?.("");
  } catch {
    /* cosmetic */
  }
  try {
    return await askStudentInner(ctx, title, options);
  } finally {
    quietForPicker = false;
  }
}

async function askStudentInner(
  ctx: any,
  title: string,
  options: string[],
): Promise<{ choice: string | null; typed: string; asked: boolean }> {
  const choice = await ctx.ui.select(`${title}   ↑↓ then Enter`, [...options, TYPE_IT]);
  if (choice !== TYPE_IT) return { choice: choice ?? null, typed: "", asked: true };
  let typed = "";
  try {
    typed = String((await ctx.ui.input?.("Type it here, then Enter:", "")) ?? "").trim();
  } catch {
    /* an older pi without ui.input: falls through as an empty typed answer */
  }
  return { choice: TYPE_IT, typed, asked: true };
}

/**
 * Shared quiet renderers: student sees a status line and a checkmark.
 *
 * `fallback` is what the line says when the model sends no `status`. It used
 * to be one generic sentence for every tool, which was fine while `status` was
 * required — but required meant a model that forgot it lost the whole call, so
 * it is optional now, and a small model promptly stopped sending it at all: a
 * live run showed "Working in the notebook…" four times in a row where the
 * student had been getting "Noting your experience level…". Optional keeps the
 * checkpoint; a per-tool fallback keeps the warmth.
 */
const quiet = (fallback: string) => ({
  renderCall(args: { status?: string }, theme: any) {
    const status =
      // .trim(): a status of a single space passed the length test and
      // printed a bare "📝  ✓" at the student.
      typeof args?.status === "string" && args.status.trim().length > 0
        ? args.status.trim()
        : fallback;
    return new Text(theme.fg("accent", `📝 ${status}`), 0, 0);
  },
  renderResult(result: any, { expanded, isPartial }: any, theme: any) {
    if (isPartial) return new Text(theme.fg("muted", "…"), 0, 0);
    // `details` is set by every path through execute(). Its ABSENCE means
    // execute never ran — pi rejected the arguments against the schema first —
    // and that used to render as a green ✓ while nothing had happened, with
    // the model's whole rejected payload (question, verbatim answer, note
    // markdown) printed into the student's terminal beside it.
    if (result?.details?.failed === true || result?.isError === true || !result?.details) {
      return new Text(theme.fg("error", "⚠ something hiccuped — your tutor is on it"), 0, 0);
    }
    const raw =
      result?.content
        ?.filter((c: any) => c.type === "text")
        .map((c: any) => c.text ?? "")
        .join("\n")
        .trim() ?? "";
    if (expanded && raw) {
      return new Text(theme.fg("success", "✓") + "\n" + theme.fg("dim", raw), 0, 0);
    }
    let line = theme.fg("success", "✓");
    if (raw && raw !== "(ok)") line += " " + theme.fg("dim", `(${keyHint("app.tools.expand", "for details")})`);
    return new Text(line, 0, 0);
  },
});

/** The generic pair, for tools with nothing more specific to say. */
const quietRender = quiet("Working in the notebook…");

// Optional on purpose. It is a nicety — one friendly line above a spinner —
// and renderCall already has a default for it, but as a required field a model
// that forgot it lost the whole call: pi rejects the arguments before execute
// runs, so the build never happens, the student watches the rejected payload
// scroll past, and the tutor apologises for a field it cannot name. Nothing
// that only decorates the screen should be able to fail a checkpoint.
const STATUS_PARAM = Type.Optional(
  Type.String({
    description:
      "ALWAYS send this. Short student-facing status in plain, friendly words, e.g. " +
      "'Preparing our next step…'. It is the only thing on their screen while the call " +
      "runs, so a specific one ('Noting how you like to learn…') is worth far more than " +
      "the generic line they get without it. No technical terms, no cell/code/error " +
      "talk — and NEVER a fact the checkpoint you are on is asking them to find (it is " +
      "shown at the moment you build that checkpoint).",
  }),
);

// ── Chapter orchestration (the deterministic "lead agent") ──────────────────
// The lesson is split into chapters (lesson/index.json). The tutor holds only
// the CURRENT chapter's script in context; chapter_done builds a handoff
// brief, injects the next script, and trims the old conversation via
// compaction — same session, same visible transcript, fresh LLM context.
type Chapter = {
  id: string;
  file: string;
  title: string;
  /** Lecture-note prose rendered under the chapter heading (see chapterOpening). */
  opening?: string;
  checkpoints: string[];
};

function loadChapters(): Chapter[] {
  try {
    const raw = fs.readFileSync(path.join(process.cwd(), "lesson", "index.json"), "utf-8");
    return (JSON.parse(raw).chapters ?? []) as Chapter[];
  } catch {
    return [];
  }
}

/** The module's own id, from lesson/index.json ("m01-euler-tour"). */
function moduleId(): string {
  try {
    const raw = fs.readFileSync(path.join(process.cwd(), "lesson", "index.json"), "utf-8");
    return String(JSON.parse(raw).module ?? "");
  } catch {
    return "";
  }
}

/** Flat checkpoint order across all chapters — the script is the authority. */
function checkpointOrder(): string[] {
  return loadChapters().flatMap((c) => c.checkpoints);
}

/** "cp2_distance_extra" (an improvised practice round) → "cp2_distance". */
function baseCheckpointId(id: string): string {
  // Repeated, because a looping tutor produced "cp0_welcome_extra_extra" —
  // one strip left an id no lookup could match, which silently disabled the
  // ordering guard and the note skeleton for every later round.
  let out = id;
  for (let i = 0; i < 8; i++) {
    const next = out.replace(/_extra(_?\d+)?$/, "");
    if (next === out) break;
    out = next;
  }
  return out;
}

function isScriptedCheckpoint(id: string): boolean {
  return checkpointOrder().includes(baseCheckpointId(id));
}

/**
 * Pull a checkpoint id back onto the script when the model has drifted near it.
 *
 * `judgment` is validated against a list and `student_response` against empty,
 * but `id` — the key everything else is looked up by — was taken as given. One
 * character out (`cp3-clustering` for `cp3_clustering`) and four things fail at
 * once, none of them visibly: the note skeleton is not found, so the
 * instructor's note cell never reaches the keepsake; the build guard and the
 * ordering guard stop recognising the checkpoint; and the closing summary
 * reports "14 of 15" while the notebook shows an answer under a misspelled
 * heading. A weak model produces exactly this kind of drift, and
 * baseCheckpointId's own comment records a real one (`cp0_welcome_extra_extra`).
 *
 * Snapping beats refusing: the tutor has already asked the question and heard
 * the answer, and a refusal it cannot satisfy costs the student the row. Only
 * near misses snap — case, the -/_ confusion, and one edit away — so an id for
 * a genuinely different checkpoint still falls through to the caller's refusal.
 */
function snapCheckpointId(id: string): { id: string; snappedFrom?: string } {
  const base = baseCheckpointId(id);
  const order = checkpointOrder();
  if (order.includes(base)) return { id };
  const suffix = id.slice(base.length); // "_extra", "_extra2", ""
  const norm = (s: string) => s.toLowerCase().replace(/[-\s.]/g, "_");
  const target = norm(base);
  let hit = order.find((c) => norm(c) === target);
  if (!hit) {
    // One edit away (a dropped, doubled or swapped character), and only when
    // exactly one candidate is that close — an ambiguous near miss is not a
    // near miss.
    const within1 = order.filter((c) => editDistanceAtMost(norm(c), target, 1));
    if (within1.length === 1) hit = within1[0];
  }
  return hit ? { id: hit + suffix, snappedFrom: id } : { id };
}

/** True when `a` and `b` differ by at most `max` insertions/deletions/substitutions. */
function editDistanceAtMost(a: string, b: string, max: number): boolean {
  if (Math.abs(a.length - b.length) > max) return false;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] : 1 + Math.min(prev[j - 1], prev[j], cur[j - 1]);
      best = Math.min(best, cur[j]);
    }
    if (best > max) return false; // no cell in this row can still win
    prev = cur;
  }
  return prev[b.length] <= max;
}

/** The checkpoint the tutor is expected to work next, or null if unknown. */
/**
 * Where to go after closing `id`: the next checkpoint in the script, full
 * stop.
 *
 * This was briefly made gap-aware — "the first checkpoint with no log row" —
 * so that a session killed mid-module could resume into the hole it left.
 * Three rounds of review found three different Blockers in that, all the
 * same shape: the ordering guard, `chapter_done`'s re-arm, the per-chapter
 * scripts and this function each have their own idea of "next", and making
 * one of them gap-aware puts it at odds with the other three — a build the
 * guard refuses, a chapter that can never be re-loaded, a walk back through
 * checkpoints that already have rows and notes, duplicate rows in the graded
 * record. Positional is the only rule all four share.
 *
 * A gap is also close to unreachable now: the build guard refuses a
 * checkpoint ahead of the open one, and this advances one at a time. The
 * gapped logs in session_artifacts/ predate both. When one does appear, the
 * honest outcome is a record that says so — the closing summary already
 * prints "Checkpoints completed: 11 of 12" and names where to pick up —
 * not a record that quietly re-runs half the module to paper over it.
 */
function nextCheckpointId(id: string): string | null {
  const order = checkpointOrder();
  const i = order.indexOf(baseCheckpointId(id));
  return i >= 0 ? (order[i + 1] ?? null) : null;
}

/**
 * Is `cpId` a checkpoint the tutor has not reached yet? This — not mere
 * inequality — is what the build-ordering guard must refuse: inserting a
 * LATER checkpoint's build while the current one is open strands its note
 * cell after that build. Re-inserting an EARLIER one (a reveal figure added
 * after the checkpoint was closed) is harmless and must stay allowed.
 */
function isAheadOf(cpId: string, openId: string): boolean {
  const order = checkpointOrder();
  const a = order.indexOf(cpId);
  const b = order.indexOf(openId);
  return a >= 0 && b >= 0 && a > b;
}

function chapterScriptMessage(ch: Chapter, num: number, total: number): string {
  const src = fs.readFileSync(path.join(process.cwd(), "lesson", ch.file), "utf-8");
  const last = ch.checkpoints[ch.checkpoints.length - 1];
  return (
    `CHAPTER SCRIPT ${num}/${total} — "${ch.title}" (invisible to the student). ` +
    `This is your curriculum right now:\n\n${src}\n\n` +
    (chapterOpening(ch)
      ? `The chapter's opening paragraph is ALREADY in their notebook, above ` +
        `your first build — do not read it out or paraphrase it. It says:\n` +
        `"${chapterOpening(ch)}"\n\n`
      : "") +
    `Work its checkpoints in order, ending each one with checkpoint_done. ` +
    `After checkpoint_done for the final checkpoint (${last}), call chapter_done ` +
    `with short handoff notes.`
  );
}

/**
 * The chapter's instructor-authored `opening:` prose, from the `chapter:`
 * block at the top of its script. This is what turns the notebook from a
 * pile of experiments into a lecture note: a heading alone tells a cold
 * reader nothing about why the next four cells exist.
 *
 * It renders BEFORE the chapter's first question, so an opening that
 * answers one is an answer leak — the scripts carry a comment saying so.
 */
function chapterOpening(ch: Chapter): string {
  return String(ch.opening ?? "").trim();
}

/**
 * Deterministic notebook structure: a "## Chapter N — Title" markdown cell
 * plus the chapter's opening prose, at every chapter start, so the finished
 * notebook reads as a document the student can re-learn from.
 * Skip-if-exists; cosmetic — never blocks.
 */
async function insertChapterHeader(
  ch: Chapter,
  num: number,
  total: number,
  signal?: AbortSignal,
): Promise<boolean> {
  try {
    const warm = await ensureWarm(signal);
    if (warm) return false;
    const name = `${ch.id}_header`;
    const opening = chapterOpening(ch);
    const heading = `## Chapter ${num} of ${total} — ${ch.title}`;
    const body = `mo.md(${pyMd(opening ? `${heading}\n\n${opening}` : heading)})`;
    await runKernel(
      `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _names = [c.name for c in ctx.cells]\n` +
        `    if ${py(name)} not in _names:\n` +
        `        _cid = ctx.create_cell(${py(body)}, name=${py(name)}, hide_code=True)\n` +
        `        ctx.run_cell(_cid)\n` +
        focusCellCode("_cid", "        "),
      signal,
    );
    await pinAppealToBottom(signal);
    return true;
  } catch {
    // headers are cosmetic — never block the lesson
    return false;
  }
}

/**
 * Chapter 1's header is the one nobody else retries: chapters 2-5 get theirs
 * from chapter_done, long after the kernel is warm, but chapter 1's fires
 * once at session start — and the kernel stays cold until a browser client
 * connects. A student who opens the page late (the D7 notebook-down path)
 * used to lose that header for good, leaving a notebook whose first heading
 * says "Chapter 2 of 5". Keep trying until it lands, the chapter moves on,
 * or we give up after ~5 minutes. create_cell is skip-if-exists, so a late
 * success can never duplicate it.
 */
/**
 * No named cell may land before its chapter header. The scheduled attempt is
 * on a timer, and cp0_welcome builds nothing — so its note cell (a greeting
 * and one dialog later) could be created first and push "Chapter 1 of 5"
 * below it. Called before every insert that creates a named cell; skip-if-
 * exists makes it a no-op after the first success.
 */
async function ensureChapterHeader(signal?: AbortSignal): Promise<void> {
  const chapters = loadChapters();
  if (chapters.length === 0) return;
  const id = currentChapterId() ?? chapters[0].id;
  const idx = chapters.findIndex((c) => c.id === id);
  if (idx < 0) return;
  await insertChapterHeader(chapters[idx], idx + 1, chapters.length, signal);
}

function scheduleChapterHeader(ch: Chapter, num: number, total: number): void {
  let attempts = 0;
  const tick = async () => {
    attempts += 1;
    if (currentChapterId() !== ch.id) return;
    const ok = await insertChapterHeader(ch, num, total);
    if (ok || attempts >= 20) return;
    const t = setTimeout(() => void tick(), 15_000);
    (t as any).unref?.();
  };
  const first = setTimeout(() => void tick(), 15_000);
  (first as any).unref?.();
}

function chapterStatePath(): string {
  return path.join(process.cwd(), "session_artifacts", "chapter_state.json");
}

function currentChapterId(): string | null {
  try {
    return JSON.parse(fs.readFileSync(chapterStatePath(), "utf-8")).current ?? null;
  } catch {
    return null;
  }
}

function writeChapterState(id: string) {
  try {
    fs.mkdirSync(path.dirname(chapterStatePath()), { recursive: true });
    fs.writeFileSync(chapterStatePath(), JSON.stringify({ current: id }));
  } catch {
    // best-effort
  }
}

function readSessionLog(): any[] {
  try {
    return fs
      .readFileSync(path.join(process.cwd(), "session_artifacts", "session_log.jsonl"), "utf-8")
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => {
        try {
          return JSON.parse(l);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

function progressBrief(entries: any[]): string {
  const cps = entries.filter((e) => e?.type === "checkpoint" && e.id);
  if (cps.length === 0) return "(no checkpoints logged yet)";
  return cps
    .map(
      (e) =>
        `${e.id}: ${e.judgment ?? "?"}` +
        (e.student_response ? ` — "${String(e.student_response).slice(0, 120)}"` : ""),
    )
    .join("\n");
}

function sessionLogPath(): string {
  return path.join(process.cwd(), "session_artifacts", "session_log.jsonl");
}

/** The tutor no longer hand-writes JSON: the extension owns the graded log. */
function appendLog(entry: Record<string, unknown>): boolean {
  try {
    fs.mkdirSync(path.dirname(sessionLogPath()), { recursive: true });
    fs.appendFileSync(
      sessionLogPath(),
      JSON.stringify({ ts: new Date().toISOString(), ...entry }) + "\n",
    );
    return true;
  } catch {
    return false;
  }
}

/**
 * Ground truth for the graded artifact: the student's own messages, read
 * straight from the transcript, so a logged answer can never drift into
 * paraphrase. Returns what they typed since the previous checkpoint.
 */
let studentSaidMark = 0;
// Every pi.sendMessage injection lands in the transcript with role "user".
// Anything added here MUST start with one of these prefixes, or it is filed
// as the student's own words and quoted back at them in the graded record.
const INJECTED_PREFIX =
  /^(CHAPTER SCRIPT|RESUME CONTEXT|=== TUTORING HANDOFF|The student clicked|Please start the tutoring session|── Chapter |NOTE \(invisible to the student\)|REFEREE VERDICT)/;

/**
 * Curriculum prose that must never be filed as something the student said.
 *
 * The prefix test above catches an injected message by how it STARTS, which
 * is enough while the message arrives whole. It did not hold: a graded row
 * in this repo carries 44 words of chapter-opening prose as the student's
 * first verbatim utterance, quoted back at them inside their own note cell
 * under "What I said we could throw away". The opening reaches the model
 * nested inside the CHAPTER SCRIPT message (chapterScriptMessage quotes it,
 * so the tutor does not read it aloud), and something between there and
 * getBranch() — compaction is the suspect — hands it back as a bare user
 * turn with the prefix gone.
 *
 * Rather than guess at the path, refuse the CONTENT. A student does not type
 * their own chapter opening word for word, so an exact match is safe, and it
 * closes the hole whatever re-wraps the message. Memoised per chapter file:
 * this runs once per captured message.
 */
let scriptProseCache: { key: string; set: Set<string> } | null = null;
function scriptProse(): Set<string> {
  const key = currentChapterId() ?? "";
  if (scriptProseCache?.key === key) return scriptProseCache.set;
  const set = new Set<string>();
  try {
    for (const ch of loadChapters()) {
      const op = chapterOpening(ch);
      // Openings only. Chapter TITLES were in here for one draft, and one of
      // them is the single word "Abstraction" — which is exactly what a
      // student may type as their answer at cp2_abstraction. A guard against
      // losing the student's words must not be the thing that loses them.
      if (op && op.length > 80) set.add(op);
    }
  } catch {
    // no chapters is not a reason to lose the student's words
  }
  scriptProseCache = { key, set };
  return set;
}

function allStudentMessages(ctx: any): string[] {
  const entries: any[] = ctx?.sessionManager?.getBranch?.() ?? [];
  const out: string[] = [];
  const prose = scriptProse();
  for (const e of entries) {
    if (e?.type !== "message" || e?.message?.role !== "user") continue;
    const c = e.message.content;
    const text =
      typeof c === "string"
        ? c
        : Array.isArray(c)
          ? c
              .filter((p: any) => p?.type === "text" && typeof p.text === "string")
              .map((p: any) => p.text)
              .join("\n")
          : "";
    const s = text.trim();
    if (!s || INJECTED_PREFIX.test(s) || prose.has(s)) continue;
    out.push(s);
  }
  return out;
}

/**
 * The question the tutor asked and never got an answer to, or "".
 *
 * `checkpoint_done` opens a dialog, and a dialog takes over the keyboard: a
 * live run ended its reveal with "what would you expect on a ring of 800
 * dots?" and closed the checkpoint in the same breath. The student's typed
 * answer went into the dialog instead of the conversation, vanished from the
 * transcript entirely, and the picker resolved itself to READY — so the
 * pacing gate the student was supposed to hold passed without them.
 */
function tutorAwaitingAnswer(ctx: any): string {
  try {
    const entries: any[] = ctx?.sessionManager?.getBranch?.() ?? [];
    for (let i = entries.length - 1; i >= 0; i--) {
      const e = entries[i];
      if (e?.type !== "message") continue;
      const role = e?.message?.role;
      if (role !== "user" && role !== "assistant") continue;
      const c = e.message.content;
      const text = (
        typeof c === "string"
          ? c
          : Array.isArray(c)
            ? c
                .filter((p: any) => p?.type === "text" && typeof p.text === "string")
                .map((p: any) => p.text)
                .join("\n")
            : ""
      ).trim();
      // A tool-only assistant turn is not the tutor speaking.
      if (!text) continue;
      if (role === "user") return "";
      // The LAST FEW lines, not only the last one. A question followed by a
      // closing remark is the commonest shape the tutor produces — ch1's
      // reveal_after scripts it that way in so many words ("THEN one typed
      // follow-up … Bridge: …") — and looking only at the final line meant the
      // guard went quiet on exactly the turn it exists for: the picker opened
      // over an unanswered question, the student's typed reply went into it,
      // and the checkpoint was logged with nothing of theirs to quote.
      for (const last of text
        .split(/\n+/)
        .filter(Boolean)
        .map((l: string) => l.trim())
        .slice(-3)
        .reverse()) {
        // Not only a trailing "?": a live run asked "does that mean your camera's
        // working now?" and then added "Just so I know for the next paper page."
        // — so the line ended in a full stop, the guard stayed quiet, and the
        // picker ate the answer exactly as before. Take the last sentence that
        // IS a question.
        // A question the tutor QUOTES is not a question it is waiting on:
        // `You wrote "is it 6?" earlier.` fired the guard and reported the
        // fragment `you just asked "You wrote "`. Mask quoted spans (same
        // length, so offsets still line up) before looking for the "?".
        const masked = last.replace(/[\u201C"][^\u201D"]*[\u201D"]/g, (m) => " ".repeat(m.length));
        if (!masked.includes("?")) continue;
        const end = masked.lastIndexOf("?") + 1;
        const upTo = last.slice(0, end);
        const parts = masked.slice(0, end - 1).split(/(?<=[.!?])\s+/);
        const startAt = parts.length > 1 ? upTo.length - parts[parts.length - 1].length - 1 : 0;
        return upTo.slice(Math.max(0, startAt)).trim();
      }
      return "";
    }
  } catch {
    /* a transcript we cannot read never blocks a close */
  }
  return "";
}

/**
 * A picker answer never becomes a transcript message, so until now the graded
 * record had NO independent evidence of it — and a live run logged
 * "Comfortable with Python" for a student who had picked "I don't code",
 * miscalibrating the whole session and putting the opposite of their answer
 * in the artifact they submit. These are captured straight off the
 * ask_user_question tool result instead.
 */
const pickedAnswers: string[] = [];
let pickedMark = 0;
/**
 * Set while the resume brief is in flight. The continue-or-fresh dialog is
 * session mechanics, not an answer to a lesson question, so its result is
 * dropped instead of riding into the next checkpoint's `student_picked`.
 */
let awaitingResumeChoice = false;

/**
 * A sentence the DIALOG produced, not the student. The package has more
 * than one: "User declined to answer questions" when it is dismissed, and
 * "User wants to chat about this. Continue the conversation to help them
 * decide." when the student takes the chat row, which sits in every
 * single-select nav cycle. Stored, either printed in the submitted record
 * as the student's own *You chose:* line — and the chat one also made the
 * picker check refuse twice and stamp a false quoting warning on the row.
 * Every one of them opens with "User ", which no option label here does.
 */
const isDialogSentinel = (s: string): boolean =>
  /^user (declined to answer|wants to chat)/i.test(s.trim()) ||
  /^\(no input\)$/i.test(s.trim());

function recordPickedAnswer(event: any): void {
  try {
    if (!/ask.?user.?question/i.test(String(event?.toolName ?? ""))) return;

    // The package supplies the answers structured on the tool result; the
    // envelope sentence is only a fallback. Parsing prose truncated an
    // answer that contained its own quotation marks.
    const structured = (event?.details?.answers ?? [])
      .map((a: any) => String(a?.answer ?? a?.value ?? "").trim())
      .filter((v: string) => v && !isDialogSentinel(v));
    const text = (event?.content ?? [])
      .filter((c: any) => c?.type === "text" && typeof c.text === "string")
      .map((c: any) => c.text)
      .join("\n")
      .trim();
    if (structured.length) {
      const fromDetails = structured.join(" · ");
      if (
        !(awaitingResumeChoice && /continue|fresh|left off|pick (things )?up/i.test(fromDetails))
      ) {
        pickedAnswers.push(fromDetails.slice(0, 300));
      } else {
        awaitingResumeChoice = false;
      }
      return;
    }
    if (!text) return;
    let picked = text;
    try {
      const parsed = JSON.parse(text);
      if (parsed?.cancelled) return;
      const answers = parsed?.answers ?? parsed;
      const flat = Array.isArray(answers)
        ? answers
        : answers && typeof answers === "object"
          ? Object.values(answers)
          : [];
      const joined = flat
        .map((a: any) => (typeof a === "string" ? a : (a?.answer ?? a?.value ?? "")))
        .filter((v: string) => v && !isDialogSentinel(v))
        .join(" · ");
      picked = joined;
    } catch {
      // Not JSON. The dialog package hands back a sentence of plumbing —
      //   User has answered your questions: "How do you feel about
      //   Python?"="tried a little". You can now continue …
      // — and storing that whole sentence put a machine's voice inside the
      // student's own `*You chose:*` line in the submitted notebook. Keep
      // only the answers: every "question"="answer" pair's right-hand side.
      // Per ANSWER, not per sentence: a dialog with two questions where one
      // was left blank must still record the one that was answered.
      // The value is matched lazily up to the quote that ends the pair (the
      // envelope puts a "." or the end of the sentence after it), so an
      // answer containing its own quotation marks survives instead of being
      // stored truncated to its first word.
      const pairs = [...text.matchAll(/"([^"]*)"\s*=\s*"([\s\S]*?)"(?=\s*[.,]|\s*$)/g)]
        .map((m) => m[2].trim())
        .filter((v) => v && !isDialogSentinel(v));
      picked = pairs.length ? pairs.join(" · ") : "";
    }
    // The continue-or-fresh answer is session mechanics, not a lesson
    // answer. Match it by CONTENT rather than "whatever dialog comes first
    // after a resume": if the tutor asks continue-or-fresh in plain text,
    // the flag would otherwise swallow the next real prediction instead —
    // and a re-asked resume dialog would slip the original one through.
    // A dismissed dialog is not an answer. The package returns a fixed
    // "User declined to answer questions", which is a machine's sentence —
    // stored, it printed in the submitted record as *You chose: "User
    // declined to answer questions"*, attributed to the student.
    //
    // And it does NOT answer the resume question, so the flag stays armed:
    // clearing it here meant the RE-ASKED continue-or-fresh answer was
    // stored instead, which is the same leak one dialog later.
    if (!picked || isDialogSentinel(picked)) return;
    if (awaitingResumeChoice && /continue|fresh|left off|pick (things )?up/i.test(picked)) {
      awaitingResumeChoice = false;
      return;
    }
    pickedAnswers.push(picked.slice(0, 300));
  } catch {
    // capture is best-effort; never break a turn
  }
}

function pickedSince(commit = true): string[] {
  const fresh = pickedAnswers.slice(pickedMark);
  if (commit) pickedMark = pickedAnswers.length;
  return fresh;
}

function studentSaidSince(ctx: any, commit = true): string[] {
  try {
    const all = allStudentMessages(ctx);
    const fresh = all.slice(studentSaidMark);
    if (commit) studentSaidMark = all.length;
    return fresh;
  } catch {
    return [];
  }
}

/**
 * The note cell's «slots» are the graded artifact's centerpiece: they must be
 * the STUDENT's words, not the tutor's prose. A live session produced
 * "A–D = 2, and the average over all 6 pairs = 7/6 ≈ 1.17" from a student who
 * had typed only "yes", "2", "7/6" — a fabricated number presented as their
 * work. So the extension checks the fills against what the student actually
 * said (transcript capture + the tutor's own verbatim field) and refuses once.
 *
 * Tolerant by design: word order, joining and connective words are free; what
 * it catches is invented content — any number they never gave, or several
 * added content words.
 */
const SLOT_GLUE = new Set([
  "a", "an", "and", "the", "of", "to", "in", "on", "at", "is", "are", "was", "were",
  "it", "its", "i", "my", "me", "we", "our", "you", "your", "that", "this", "these",
  "those", "so", "then", "for", "with", "as", "but", "or", "if", "not", "be", "been",
  "there", "here", "each", "every", "both", "than", "when", "because", "about",
  // Structural labels a tutor puts around a multi-part answer — the note
  // skeletons say «their answers, verbatim» (plural) for the 3-6 part
  // checkpoints, so "Idea: … Count: … Fraction: …" is the natural shape and
  // must not read as invented content.
  "idea", "answer", "answers", "count", "counts", "fraction", "result", "work",
  "note", "first", "second", "third", "final", "total", "corrected", "then",
]);

function slotTokens(s: string): string[] {
  const norm = s
    .toLowerCase()
    .replace(/[‐-―−]/g, "-")
    .replace(/[*_`>#]/g, " ");
  return norm.match(/[a-z0-9](?:[a-z0-9._/-]*[a-z0-9])?/g) ?? [];
}

/**
 * A "figure": 7/6, 1.17, 4.74, 0.61 — a compound or multi-digit
 * number. Bare single digits are NOT figures: they are how a subscript
 * ($L_0$ → "l", "0") and an ordinary label ("dot 1 to dot 5") tokenize, and
 * flagging those refused faithful records.
 */
function isFigure(token: string): boolean {
  return /^\d+([./:,-]\d+)+$|^\d{2,}$/.test(token);
}

/**
 * Character-bigram overlap (Dice), 0–1. Used to spot a RETYPED answer: a
 * live session logged "becuase tirangles are ipormtat" for a student who
 * typed "becuase tirangles are ipmortat" — the model copied their sentence
 * out by hand and re-scrambled their own typo along the way. Word-level
 * drift cannot see that (one odd token), and it is exactly the kind of
 * silent edit the graded record must not carry.
 */
function bigramDice(a: string, b: string): number {
  const grams = (s: string) => {
    const t = s.toLowerCase().replace(/\s+/g, " ").trim();
    const out = new Set<string>();
    for (let i = 0; i + 1 < t.length; i += 1) out.add(t.slice(i, i + 2));
    return out;
  };
  const A = grams(a);
  const B = grams(b);
  if (A.size === 0 || B.size === 0) return 0;
  let hits = 0;
  for (const g of A) if (B.has(g)) hits += 1;
  return (2 * hits) / (A.size + B.size);
}

/**
 * If `response` is a near-copy of one of the student's own messages, return
 * that message instead — their words, character for character.
 *
 * Similarity alone is not enough to act on. Joining several short answers
 * into one ("A–D is 2, and the average is 7/6 over all six pairs") is the
 * sanctioned shape — the skeletons say «their answers, verbatim», plural —
 * and it scores nearly as high against its longest fragment as a retype
 * does against the whole message. Snapping there would delete half the
 * student's answer from the graded record, so a candidate must also be
 * about the same LENGTH: a retype is, a swallowed fragment is not.
 */
function snapToTranscript(response: string, said: string[]): string | null {
  const flat = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();
  if (said.some((s) => flat(s) === flat(response) || flat(s).includes(flat(response)))) return null;
  let best: { score: number; text: string } | null = null;
  for (const s of said) {
    const score = bigramDice(s, response);
    if (!best || score > best.score) best = { score, text: s };
  }
  if (!best || best.score < 0.8) return null;
  const longest = Math.max(best.text.length, response.length);
  return Math.abs(best.text.length - response.length) <= 0.15 * longest ? best.text : null;
}

/** Content tokens in `fill` that the student never produced. */
function slotDrift(fill: string, studentPool: string[]): { numbers: string[]; words: string[] } {
  const pool = new Set(studentPool.flatMap(slotTokens));
  const numbers: string[] = [];
  const words: string[] = [];
  for (const t of new Set(slotTokens(fill))) {
    if (pool.has(t) || SLOT_GLUE.has(t)) continue;
    if (isFigure(t)) numbers.push(t);
    else words.push(t);
  }
  return { numbers, words };
}

/**
 * Pull a checkpoint's instructor-authored `note:` block out of its chapter
 * YAML (block scalar, dedented). The tutor fills «slots», not prose.
 */
// Exact id only. A practice round (`cp2_distance_extra`) is the same QUESTION
// on new DATA, so its base checkpoint's skeleton is wrong for it: cp2's says
// "$L = 7/6$", while the practice variant is a star ($L = 9/6$). The tutor
// writes note_markdown for those, per AGENTS.md.
function checkpointBlock(cpId: string, key: string): string {
  try {
    const chapter = loadChapters().find((c) => c.checkpoints.includes(cpId));
    if (!chapter) return "";
    const lines = fs
      .readFileSync(path.join(process.cwd(), "lesson", chapter.file), "utf-8")
      .split("\n");
    const idRe = new RegExp(`^\\s*-\\s+id:\\s*${cpId}\\s*$`);
    const start = lines.findIndex((l) => idRe.test(l));
    if (start < 0) return "";
    let end = lines.length;
    for (let i = start + 1; i < lines.length; i++) {
      if (/^\s*-\s+id:\s/.test(lines[i])) {
        end = i;
        break;
      }
    }
    for (let i = start; i < end; i++) {
      const m = new RegExp(`^(\\s*)${key}:\\s*\\|`).exec(lines[i]);
      if (!m) continue;
      const keyIndent = m[1].length;
      const block: string[] = [];
      for (let j = i + 1; j < end; j++) {
        const l = lines[j];
        if (!l.trim()) {
          block.push("");
          continue;
        }
        if (l.length - l.trimStart().length <= keyIndent) break;
        block.push(l);
      }
      const indents = block.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length);
      const base = indents.length ? Math.min(...indents) : 0;
      return block.map((l) => l.slice(base)).join("\n").trim();
    }
    return "";
  } catch {
    return "";
  }
}

/** A checkpoint's instructor-authored `note:` skeleton. */
function noteSkeleton(cpId: string): string {
  return checkpointBlock(cpId, "note");
}

/**
 * `note: none` in the script — this checkpoint gets NO note cell. Some
 * checkpoints are session mechanics, not lecture: cp0 calibrates how much
 * code the student wants to see, and a cell about that is the first thing a
 * cold reader meets in a document that has not begun. The answer still gets
 * logged and still lands in the closing session_record.
 */
function noteSuppressed(cpId: string): boolean {
  try {
    const chapter = loadChapters().find((c) => c.checkpoints.includes(baseCheckpointId(cpId)));
    if (!chapter) return false;
    const lines = fs
      .readFileSync(path.join(process.cwd(), "lesson", chapter.file), "utf-8")
      .split("\n");
    const start = lines.findIndex((l) =>
      new RegExp(`^\\s*-\\s+id:\\s*${baseCheckpointId(cpId)}\\s*$`).test(l),
    );
    if (start < 0) return false;
    for (let i = start + 1; i < lines.length; i++) {
      if (/^\s*-\s+id:\s/.test(lines[i])) break;
      if (/^\s*note:\s*none\s*$/.test(lines[i])) return true;
    }
    return false;
  } catch {
    return false;
  }
}

/** Does the script offer a practice variant for this checkpoint? */
function hasFreshVariants(cpId: string): boolean {
  return checkpointBlock(cpId, "fresh_variants").trim().length > 0;
}

/**
 * The cell names a checkpoint's `build:` promises — every
 * `nb_add_template("X")` in it, resolved through `cells/X.py`'s
 * `# --- cell: NAME ---` markers.
 *
 * Used to catch a build the tutor simply never did. cp5_ring_formula asks
 * seven things in sequence and the seventh is "photograph that page"; in a
 * live run the model reached the end of the reasoning, decided the answer
 * was complete, and closed the checkpoint — the upload area was never
 * inserted and the student was never asked for the photo at all. The
 * hand-worked page is the point of those checkpoints, so a promise in the
 * script is checked rather than trusted.
 */
function scriptedBuildCells(cpId: string): string[] {
  const build = checkpointBlock(cpId, "build");
  if (!build.trim() || /^\s*none\s*$/i.test(build)) return [];
  const out: string[] = [];
  // Tolerate the argument form the tool itself asks for —
  // nb_add_template("x", checkpoint="cp4") — and a named first argument.
  // Requiring the bare one-arg call meant an instructor aligning a build:
  // line with the tool contract silently switched this guard off.
  for (const m of build.matchAll(/nb_add_template\(\s*(?:template\s*=\s*)?["']([\w-]+)["']/g)) {
    try {
      const src = fs.readFileSync(path.join(process.cwd(), "cells", `${m[1]}.py`), "utf-8");
      for (const c of src.matchAll(/^#\s*---\s*cell:\s*(\w+)\s*---/gm)) out.push(c[1]);
    } catch {
      // an unknown template name is nb_add_template's problem, not ours
    }
  }
  return out;
}

/**
 * Of a checkpoint's build cells, the ones that actually ask for a photograph.
 *
 * A cell holds a drop area when it CREATES one — `mo.ui.file(...)` — which is
 * the same test nb_add_template uses to decide whether to tell the tutor to
 * ask for the page. The photo gate below used to look at the cell's NAME
 * instead, and any name ending in `_photo` armed it. m01's chapter 3 reveals
 * the 1944 bombing with a historical plate in a cell called
 * `cp3_bombed_photo`: display-only, nothing to upload, and no ask anywhere in
 * the script. Closing that checkpoint was refused with "no photo has reached
 * me", and the second attempt logged it stamped `photo_missing`, which the
 * closing summary prints as a page the student never took. Two guards reading
 * the same build disagreed about what a photo cell is; now they run the same
 * test.
 */
function scriptedPhotoCells(cpId: string): string[] {
  const build = checkpointBlock(cpId, "build");
  if (!build.trim() || /^\s*none\s*$/i.test(build)) return [];
  const out: string[] = [];
  for (const m of build.matchAll(/nb_add_template\(\s*(?:template\s*=\s*)?["']([\w-]+)["']/g)) {
    try {
      const src = fs.readFileSync(path.join(process.cwd(), "cells", `${m[1]}.py`), "utf-8");
      // Split on the cell markers, keeping each name with the body under it.
      const parts = src.split(/^#\s*---\s*cell:\s*(\w+)\s*---$/m);
      for (let i = 1; i < parts.length; i += 2) {
        if (/\bmo\.ui\.file\s*\(/.test(parts[i + 1] ?? "")) out.push(parts[i]);
      }
    } catch {
      // an unknown template name is nb_add_template's problem, not ours
    }
  }
  return out;
}

/**
 * How many questions the checkpoint's own script asks, counted from the
 * numbered list in its `ask` block ("1." … "2." …). Zero when the block does
 * not number them, which switches the check that uses this off rather than
 * guessing.
 */
function scriptedQuestionCount(cpId: string): number {
  return (checkpointBlock(cpId, "ask").match(/^\s*\d+\.\s/gm) ?? []).length;
}

/** The «…» markers of a note skeleton, in order. */
function slotMarkers(skeleton: string): string[] {
  return skeleton.match(/«[^»]*»/g) ?? [];
}

/**
 * Fill a note skeleton's «slots» in order.
 *
 * An unsupplied slot falls back to `student_response` — but only the FIRST
 * one. Padding every empty slot with it printed the same sentence under
 * three different labels once the skeletons were split into a slot per part
 * ("The ring world: …", "The random world: <same sentence>", "Which one I
 * live in: <same sentence>"), which reads as the student saying something
 * they never said. checkpoint_done nudges for the missing fills before it
 * ever gets here; this is the shape when it has nudged twice and given up.
 */
/**
 * Words that can only ever be a student clearing their throat. A «verbatim»
 * note slot quotes what they typed since the last checkpoint, and a live
 * stream is not all answer: "yep im ready" to a pace question, "ohh ok got
 * it" after a hint, and the student's OWN detour question all landed in the
 * keepsake as their worked answer — the more they engaged, the more polluted
 * their note. Lesson answers are never made ONLY of these, and the one that
 * could be ("right", "sure") is rescued by the student_response check below.
 */
const ACK_WORDS = new Set(
  ("ok okay k kk yep yup yeah ya sure right true ready im i m ive got it gotcha see ah oh ohh " +
    "hm hmm mm uh huh thanks thanku thank you cool nice great awesome perfect makes sense " +
    "understood understand lets go going next continue done finished fine alright allright " +
    "sounds good please well so and then ill let s " +
    // normMsg strips the apostrophe, so "let's" arrives as "lets" and never
    // matched the "let" + "s" pair above. With the words that travel with it:
    // "yeah lets keep going" answered the tutor's "ready to move on?" after a
    // detour and was filed in the next checkpoint's note as the student's
    // worked answer, between two real ones. Every word must be an ack word
    // for a message to be dropped, so these cannot swallow an answer alone.
    "lets keep carry on move moving ahead forward " +
    // A whole turn answering the tutor's own offer — "yeah, a quick note would
    // be nice" — is not the student's work, and a live run left it sitting in
    // the note between two real answers. `yes` and `no` stay OUT: they can be
    // the answer to a lesson question.
    // `this`, `that`, `one` are NOT here: "this one", answering "which of the
    // two worlds do you live in?", is half an answer, and adding them deleted
    // it from the keepsake while the student's reason survived.
    "would could should will do does did a an the be is are quick note maybe bit little")
    .split(" "),
);
const normMsg = (m: string) =>
  m
    .toLowerCase()
    .replace(/['\u2018\u2019\u02BC]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
/** True for a message that is acknowledgement and nothing else. */
function isFillerMessage(m: string): boolean {
  const t = normMsg(m);
  if (!t || /\d/.test(t)) return false;
  const words = t.split(" ");
  // Eight, not five: the live example ran seven. Still short enough that a
  // real answer is safe — and one that is not is rescued by matching
  // student_response.
  if (words.length > 8) return false;
  return words.every((w) => ACK_WORDS.has(w));
}

/**
 * Questions the student asked that are already recorded as detours. Their own
 * question is not their answer, and log_detour deliberately peeks at the
 * transcript without consuming it, so without this the question is quoted
 * twice: once in the souvenir, once in the next checkpoint's note as work.
 * Cleared when the checkpoint that follows them closes.
 */
const detourAsked = new Set<string>();

function fillSlots(skeleton: string, slots: string[], fallback: string): string {
  let i = 0;
  let usedFallback = false;
  return skeleton.replace(/«[^»]*»/g, () => {
    const supplied = slots[i];
    i += 1;
    // `.trim()`, matching the slot-count guard: a whitespace pad satisfied
    // neither, and rendered as a heading with nothing under it.
    if (supplied !== undefined && supplied.trim() !== "") return supplied;
    // The fallback is for a ONE-slot skeleton. On a skeleton whose other slot
    // already quotes the student, using it again printed their sentence twice
    // in one line: "**My guess:** way off, i said 20 — "way off, i said 20"".
    if (!fallback.trim() || usedFallback) return "*(not answered)*";
    usedFallback = true;
    return fallback;
  });
}

/**
 * A souvenir cell opens with the student's own question, quoted. Every live
 * session so far produced detour cells that answered a question the notebook
 * never states — unreadable months later, and the personalization is the
 * whole point of a souvenir. So the extension puts the quote there itself
 * rather than trusting the model to remember, and skips it when the markdown
 * already carries the question. The line goes under a leading heading if
 * there is one, so "### 🧭 Detour: …" stays first.
 */
function withQuotedQuestion(markdown: string, question: string, alsoQuoted = ""): string {
  if (!question) return markdown;
  // Words, not bytes — and the log_detour gap check normalises the same
  // way, or the two halves of the contract disagree: a cell that quotes
  // "can't" against a question typed "cant" satisfies that check and then
  // gets a SECOND copy of the question prepended by this one.
  const flat = (s: string) =>
    s
      .toLowerCase()
      .replace(/['\u2018\u2019\u02BC"\u201C\u201D]/g, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  // `question` may have been upgraded to the student's fuller message after
  // the model authored this cell, so the model's own wording counts as
  // already-quoted too. Missing that put "You asked" in the cell twice.
  if (flat(markdown).includes(flat(question))) return markdown;
  if (alsoQuoted && flat(markdown).includes(flat(alsoQuoted))) return markdown;
  const quote = `> 🧭 **You asked:** “${question}”`;
  const lines = markdown.split("\n");
  if (/^\s*#{1,6}\s/.test(lines[0] ?? "")) {
    return [lines[0], "", quote, ...lines.slice(1)].join("\n");
  }
  return `${quote}\n\n${markdown}`;
}

/** Names of every cell currently in the notebook, or null if unreadable. */
async function notebookCellNames(signal?: AbortSignal): Promise<string[] | null> {
  const r = await runKernel(
    `import marimo._code_mode as cm\n` +
      `async with cm.get_context() as ctx:\n` +
      `    print("CELLS<<" + ",".join(str(getattr(c, "name", "")) for c in ctx.cells) + ">>")\n`,
    signal,
  );
  if (r.failed) return null;
  const m = /CELLS<<(.*)>>/.exec(r.out);
  return m ? m[1].split(",").filter(Boolean) : null;
}

/**
 * The source of a named cell, or null when it cannot be read (kernel cold,
 * marimo API shifted). Used to CHECK a souvenir the tutor built itself —
 * the `cell_name` path was the unchecked half of the contract, and it is
 * the path the model actually takes.
 */
async function readCellSource(name: string, signal?: AbortSignal): Promise<string | null> {
  const r = await runKernel(
    `import marimo._code_mode as cm\n` +
      `async with cm.get_context() as ctx:\n` +
      `    _c = None\n` +
      `    for _x in ctx.cells:\n` +
      `        if getattr(_x, "name", None) == ${py(name)}:\n` +
      `            _c = _x\n` +
      `    if _c is None:\n` +
      `        print("SOUVENIR_MISSING")\n` +
      `    else:\n` +
      `        _src = ""\n` +
      `        for _a in ("code", "source", "text"):\n` +
      `            _v = getattr(_c, _a, None)\n` +
      `            if isinstance(_v, str) and _v.strip():\n` +
      `                _src = _v\n` +
      `                break\n` +
      `        print("SOUVENIR_SRC<<")\n` +
      `        print(_src)\n` +
      `        print(">>SOUVENIR_SRC")\n`,
    signal,
  );
  if (r.failed) return null;
  if (r.out.includes("SOUVENIR_MISSING")) return "";
  const m = /SOUVENIR_SRC<<\n([\s\S]*?)\n>>SOUVENIR_SRC/.exec(r.out);
  const src = m?.[1] ?? "";
  // Empty means the attribute walk found nothing — unknown, not empty. A
  // check that cannot see the cell must stand down, never accuse.
  return src.trim() ? src : null;
}

/**
 * Put the student's own question at the top of a souvenir cell the tutor
 * built. The bounce asks for it once; if the retry still paraphrases, the
 * quote is one line and we have it — better in the keepsake by machine than
 * absent forever, which is what "bounce once, then accept" left behind.
 */
async function prependQuestionToCell(
  name: string,
  question: string,
  signal?: AbortSignal,
): Promise<boolean> {
  const src = await readCellSource(name, signal);
  if (!src) return false;
  // The rewrite is done by Python, in the kernel, with `ast` — the cell's
  // last top-level statement is a thing only a parser can find. String
  // surgery on "the last unindented line" put the quote UNDER the picture
  // for a souvenir whose vstack closes on its own line, and produced
  // unparseable source for a multi-line netviz(...) call. Both are shapes
  // nb_review.py itself emits.
  const quote = `mo.md(r"""> 🧭 **You asked:** “${question.replace(/"""/g, '"')}”""")`;
  const r = await runKernel(
    `import ast\n` +
      `import marimo._code_mode as cm\n` +
      `_src = ${py(src)}\n` +
      `_quote = ${py(quote)}\n` +
      `_tree = ast.parse(_src)\n` +
      `_lines = _src.split("\\n")\n` +
      `if not _tree.body or not isinstance(_tree.body[-1], ast.Expr):\n` +
      `    print("NOT_A_DISPLAY")\n` +
      `else:\n` +
      `    _last = _tree.body[-1]\n` +
      `    _head = "\\n".join(_lines[: _last.lineno - 1])\n` +
      `    _tail = "\\n".join(_lines[_last.lineno - 1 : _last.end_lineno])\n` +
      `    _v = _last.value\n` +
      `    _is_vstack = (\n` +
      `        isinstance(_v, ast.Call)\n` +
      `        and isinstance(_v.func, ast.Attribute)\n` +
      `        and _v.func.attr == "vstack"\n` +
      `        and _v.args\n` +
      `        and isinstance(_v.args[0], ast.List)\n` +
      `    )\n` +
      `    if _is_vstack:\n` +
      `        _open = _tail.index("[") + 1\n` +
      `        _body = _tail[:_open] + "\\n    " + _quote + "," + _tail[_open:]\n` +
      `    else:\n` +
      `        _indented = "\\n".join("    " + _l for _l in _tail.split("\\n"))\n` +
      `        _body = "mo.vstack([\\n    " + _quote + ",\\n" + _indented + ",\\n])"\n` +
      `    _new = (_head + "\\n" + _body) if _head else _body\n` +
      `    ast.parse(_new)\n` +
      `    async with cm.get_context() as ctx:\n` +
      `        ctx.edit_cell(${py(name)}, _new)\n` +
      `        ctx.run_cell(${py(name)})\n` +
      `    print("QUOTED")\n`,
    signal,
  );
  return !r.failed && r.out.includes("QUOTED");
}

/**
 * Notes whose cell could not be written, kept on disk exactly as rendered.
 *
 * The tutor must never re-author one: a live outage put a chapter boundary
 * (and a context compaction) between the answer and the retry, and the
 * rebuilt note paraphrased the student in the artifact their work is graded
 * from. Parked here, the text is the same bytes whenever it lands.
 */
const parkedDir = () => path.join(process.cwd(), "session_artifacts", "parked_notes");

function parkNote(id: string, markdown: string): void {
  try {
    fs.mkdirSync(parkedDir(), { recursive: true });
    // The chapter rides in the filename: at flush, a note belonging to the
    // CURRENT chapter may create that chapter's heading, while one recovered
    // into a later chapter must not — otherwise it is filed under a chapter
    // it has nothing to do with, or lands above its own heading.
    const chapter = sanitize(currentChapterId() || "ch");
    fs.writeFileSync(path.join(parkedDir(), `${chapter}--${sanitize(id)}.md`), markdown);
  } catch {
    // best effort — the log still holds the student's words
  }
}

/** Drop a parked note once its cell is safely in the notebook. */
function unparkNote(id: string): void {
  try {
    const chapter = sanitize(currentChapterId() || "ch");
    fs.rmSync(path.join(parkedDir(), `${chapter}--${sanitize(id)}.md`), { force: true });
  } catch {
    // A leftover park file is harmless: flushParkedNotes skips a cell that is
    // already on the page, and the closing summary only lists what is left.
  }
}

/**
 * Insert every parked note, oldest first, before anything else goes in.
 * Called at the top of each build so a recovered note lands in its own
 * place rather than after the cells added while the notebook was down.
 */
async function flushParkedNotes(signal?: AbortSignal): Promise<void> {
  let files: string[];
  try {
    files = fs
      .readdirSync(parkedDir())
      .filter((f) => f.endsWith(".md"))
      // By WHEN they were parked. Sorting names put cp6_large_n_experiment
      // before cp6_watts_strogatz, which is back to front.
      .map((f) => ({ f, t: fs.statSync(path.join(parkedDir(), f)).mtimeMs }))
      .sort((a, b) => a.t - b.t)
      .map(({ f }) => f);
  } catch {
    return;
  }
  for (const f of files) {
    const full = path.join(parkedDir(), f);
    try {
      const md = fs.readFileSync(full, "utf-8");
      const [chapter, cell] = f.replace(/\.md$/, "").split("--");
      const ownChapter = !!cell && chapter === (currentChapterId() || "");
      const r = await insertMarkdownCell(cell ?? chapter, md, signal, !ownChapter);
      if (!r.failed) fs.rmSync(full, { force: true });
    } catch {
      return;
    }
  }
}

/** Insert (or skip, if present) a markdown cell and scroll the page to it. */
async function insertMarkdownCell(
  name: string,
  markdown: string,
  signal?: AbortSignal,
  skipHeader = false,
): Promise<{ out: string; failed: boolean; reason?: string }> {
  const warm = await ensureWarm(signal);
  if (warm) return warm;
  // A note recovered from an outage belongs to the chapter it was written
  // in, so it must not be the thing that creates the CURRENT chapter's
  // heading — that would file it under a chapter it has nothing to do with.
  if (name !== "session_record" && !skipHeader) await ensureChapterHeader(signal);
  // Parked notes go in before anything else, so a note recovered after an
  // outage lands in its own place rather than after the cells written while
  // the notebook was down.
  const body = `mo.md(${pyMd(markdown)})`;
  const r = await runKernel(
    `import marimo._code_mode as cm\n` +
      `async with cm.get_context() as ctx:\n` +
      `    _names = [c.name for c in ctx.cells]\n` +
      `    if ${py(name)} in _names:\n` +
      `        print("note cell already there — skipped")\n` +
      `    else:\n` +
      `        _cid = ctx.create_cell(${py(body)}, name=${py(name)}, hide_code=True)\n` +
      `        ctx.run_cell(_cid)\n` +
      focusCellCode("_cid", "        "),
    signal,
  );
  if (!r.failed) await pinAppealToBottom(signal);
  return r;
}

/**
 * Keep the ⚖️ "Tutor gets stuck" box the LAST thing on the page: every
 * insert lands above it, so the student always finds the appeal box at the
 * bottom, right under the newest material. The box's two cells are
 * anonymous on purpose (nb_fresh_start's wipe deletes every NAMED cell),
 * so they are found by their code instead of a name. Moves are visual only
 * — no cell re-executes. Purely cosmetic: a failure never blocks an insert.
 */
async function pinAppealToBottom(signal?: AbortSignal): Promise<void> {
  try {
    await runKernel(
      `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _app = [c.id for c in ctx.cells if "tutor_stuck_send" in c.code]\n` +
        `    _ids = [c.id for c in ctx.cells]\n` +
        `    if len(_app) == 2 and _ids[-2:] != _app:\n` +
        `        ctx.move_cell(_app[0], after=_ids[-1])\n` +
        `        ctx.move_cell(_app[1], after=_app[0])\n` +
        `    print("ok")\n`,
      signal,
    );
  } catch {
    /* cosmetic — never let pinning break an insert */
  }
}

/** The closing record + summary are DERIVED from the log, never retyped. */
function buildSessionRecord(entries: any[]): string {
  const cps = entries.filter((e) => e?.type === "checkpoint" && e.id);
  const detours = entries.filter((e) => e?.type === "detour");
  const lines = [
    "## 📋 Session record",
    "",
    "*Your answer to each question, and every word you typed while working on*",
    "*it — this is what gets reviewed, not the code. Hints are never held*",
    "*against you.*",
    "",
  ];
  for (const e of cps) {
    const hints = Number(e.hints_used ?? 0);
    // Two separate things, never conflated. The quote is the tutor's record
    // of the ANSWER — which for a picker choice or a drawing is the only
    // record there is. Beneath it, verbatim, goes everything the student
    // actually typed while working on this checkpoint, straight from the
    // transcript. Printing the capture AS the answer was wrong: a stray
    // "ok" typed while a dialog was open would have become their answer.
    const saidRaw: string[] = Array.isArray(e.student_said_verbatim)
      ? e.student_said_verbatim.map((s: unknown) => String(s ?? "").trim()).filter(Boolean)
      : [];
    const quote = String(e.student_response ?? "").trim();
    lines.push(
      `**${e.id}** · ${e.judgment ?? "?"}${hints ? ` · ${hints} hint${hints > 1 ? "s" : ""}` : ""}`,
      "",
      `*${String(e.question ?? "").trim()}*`,
      "",
      `> ${quote.replace(/\n+/g, " ")}`,
      "",
    );
    const pickedRaw: string[] = Array.isArray(e.student_picked)
      ? e.student_picked.map((s: unknown) => String(s ?? "").trim()).filter(Boolean)
      : [];
    if (pickedRaw.length) {
      lines.push(`*You chose:* ${pickedRaw.map((s) => `"${s}"`).join(" · ")}`, "");
    }
    if (saidRaw.length) {
      lines.push(
        `*You typed:* ${saidRaw.map((s) => `"${s.replace(/\n+/g, " ")}"`).join(" · ")}`,
        "",
      );
    }
    if (e.notes) lines.push(`*Tutor's note:* ${String(e.notes).trim()}`, "");
    // A drift the tutor was warned about twice and logged anyway belongs in
    // the submitted artifact, not only in the log the student never opens.
    if (Array.isArray(e.verbatim_drift) && e.verbatim_drift.length) {
      lines.push(
        `*⚠ Quoting check: the wording above was flagged as not matching what you typed — ` +
          `your own words are the line above it.*`,
        "",
      );
    }
  }
  const scripted = checkpointOrder();
  const haveIds = new Set(cps.map((e) => baseCheckpointId(String(e.id))));
  const furthest = scripted.reduce((acc, id, i) => (haveIds.has(id) ? i : acc), -1);
  const holes = scripted.filter((c) => !haveIds.has(c) && scripted.indexOf(c) < furthest);
  const notReached = scripted.filter((c) => !haveIds.has(c) && scripted.indexOf(c) > furthest);
  let stillParked: string[] = [];
  try {
    stillParked = fs
      .readdirSync(parkedDir())
      .filter((f) => f.endsWith(".md"))
      .map((f) => f.replace(/\.md$/, "").replace(/^[^-]*--/, ""));
  } catch {
    stillParked = [];
  }
  if (holes.length || notReached.length || stillParked.length) {
    lines.push(`### ⚠ Not everything is here`, "");
    if (stillParked.length) {
      lines.push(
        `*The notebook was unreachable when these notes were written, and it never came ` +
          `back, so they are saved beside this notebook rather than in it ` +
          `(session_artifacts/parked_notes/): ${stillParked.join(", ")}. Your answers ` +
          `themselves are in the record below.*`,
        "",
      );
    }
    if (holes.length) {
      lines.push(
        `*A session ended part-way and the next one carried on, so these have no answer ` +
          `of yours: ${holes.join(", ")}.*`,
        "",
      );
    }
    if (notReached.length) {
      lines.push(`*The module stops here — still to come: ${notReached.join(", ")}.*`, "");
    }
  }
  if (detours.length) {
    lines.push(`### 🧭 Your own questions (${detours.length})`, "");
    for (const d of detours) lines.push(`- *${String(d.question ?? "").trim()}*`);
    lines.push("");
  }
  return lines.join("\n");
}

function buildSessionSummary(entries: any[], allCheckpoints: string[]): string {
  const cps = entries.filter((e) => e?.type === "checkpoint" && e.id);
  // Count the SCRIPT's checkpoints, not the log rows: a practice round is
  // logged under its own `_extra` id, and counting those produced
  // "Checkpoints completed: 13 of 12" on the first full run — a summary the
  // grader cannot read as anything but a bug. Extra rounds are worth
  // reporting, just not as progress through the module.
  const done = new Set(cps.map((e) => baseCheckpointId(String(e.id))));
  const extras = cps.filter((e) => /_extra/.test(String(e.id))).length;
  const missing = allCheckpoints.filter((id) => !done.has(id));
  // Which curriculum produced this. A module gets edited between terms, and a
  // grader reading a submission six weeks later has no other way to tell which
  // version of a checkpoint the student actually met.
  const modId = moduleId();
  const modVer = (() => {
    try {
      const raw = fs.readFileSync(path.join(process.cwd(), "lesson", "index.json"), "utf-8");
      return String(JSON.parse(raw).version ?? "");
    } catch {
      return "";
    }
  })();
  const out = [
    "# Session summary",
    "",
    ...(modId ? [`Module: ${modId}${modVer ? ` v${modVer}` : ""}`] : []),
    `Checkpoints completed: ${allCheckpoints.filter((id) => done.has(id)).length} of ${allCheckpoints.length}`,
    ...(extras ? [`Extra practice rounds asked for: ${extras}`] : []),
    `Detours (student's own questions): ${entries.filter((e) => e?.type === "detour").length}`,
    "",
  ];
  for (const e of cps) {
    out.push(
      `## ${e.id} — ${e.judgment ?? "?"} (${Number(e.hints_used ?? 0)} hint${Number(e.hints_used ?? 0) === 1 ? "" : "s"})`,
      `Question: ${String(e.question ?? "").trim()}`,
      `Answer (verbatim): ${String(e.student_response ?? "").trim()}`,
    );
    if (Array.isArray(e.student_said_verbatim) && e.student_said_verbatim.length) {
      out.push(`Typed by the student: ${JSON.stringify(e.student_said_verbatim)}`);
    }
    if (e.notes) out.push(`Tutor's note: ${String(e.notes).trim()}`);
    // Gaps the guards gave up on. Each of these is written on the row and, up
    // to now, read by nothing: a checkpoint closed without its figure, without
    // the page it was about, or by the referee looked exactly like one that
    // went to plan. The log was honest; every view derived from it was not.
    if (Array.isArray(e.build_missing) && e.build_missing.length) {
      out.push(
        `NOT BUILT: ${e.build_missing.join(", ")} — this checkpoint's figure is ` +
          `not in the notebook (the notebook was unreachable, or the tutor never ` +
          `ran the build). The answer below was given without it.`,
      );
    }
    if (e.photo_missing) {
      out.push(`NO PHOTO: closed without the pen-and-paper page this checkpoint asks for.`);
    }
    if (e.closed_by_referee) {
      out.push(`Closed by referee ruling: ${String(e.closed_by_referee)} (the student appealed).`);
    }
    if (e.id_snapped_from) {
      out.push(
        `(the tutor sent the id as "${String(e.id_snapped_from)}"; corrected to the script's)`,
      );
    }
    // The note quotes the transcript, so a grader never has to trust the
    // tutor's memory. What is worth a human eye is what the note left out.
    if (Array.isArray(e.figures_not_quoted) && e.figures_not_quoted.length) {
      out.push(
        `Numbers they typed that no slot quotes: ${e.figures_not_quoted.join(", ")} ` +
          `(may be a self-correction or an aside — worth a glance)`,
      );
    }
    if (Array.isArray(e.note_skipped_msgs) && e.note_skipped_msgs.length) {
      out.push(
        `Typed here but NOT quoted in the note: ${e.note_skipped_msgs
          .map((n: unknown) => `msg ${n}`)
          .join(", ")} (a question they already have a souvenir for, or ` +
          `acknowledgement — the full list above is the record, and worth a ` +
          `glance if one of these looks like part of their answer)`,
      );
    }
    out.push("");
  }
  // Appeals are logged (type "appeal") and were read by neither closing
  // artifact, so a session where the student had to go over the tutor's head
  // — the one signal that says the tutor was failing them — reached the
  // grader only if they opened the raw JSONL. Using the button is
  // participation, and the record should show it as such.
  const appeals = entries.filter((e: any) => e.type === "appeal");
  if (appeals.length) {
    out.push(`## Appeals to the referee (${appeals.length})`);
    out.push(
      `The student pressed the ⚖️ box. This counts as participation, never against ` +
        `them — but it is worth reading, because it is where they felt stuck.`,
    );
    for (const a of appeals) {
      out.push(
        `- Their case: ${String(a.student_case ?? "").trim()}`,
        `  Ruling: ${String(a.ruling ?? "?")} — ${String(a.reason ?? "").trim()}` +
          (a.referee
            ? ` [${String(a.referee)}]`
            : " [referee unreachable — the tutor resolved it]"),
      );
    }
    out.push("");
  }
  try {
    const stranded = fs
      .readdirSync(parkedDir())
      .filter((f) => f.endsWith(".md"))
      .map((f) => f.replace(/\.md$/, "").replace(/^[^-]*--/, ""));
    if (stranded.length) {
      out.push(
        `## Notes that never reached the notebook`,
        `The notebook was unreachable when these were written and did not come back, so ` +
          `they are saved as markdown beside it in session_artifacts/parked_notes/: ` +
          `${stranded.join(", ")}. The answers themselves are below.`,
        "",
      );
    }
  } catch {
    // no parked notes is the normal case
  }
  if (missing.length) {
    // Name them ALL. Reporting only the first said "9 of 12" and then
    // pointed at one id, so two missing checkpoints appeared nowhere in the
    // artifact the grader reads. And a module whose last checkpoint IS
    // logged is not "where to pick up" — it is a record with holes, which
    // is what it should say.
    // Two different things, and they were being conflated: checkpoints
    // BEHIND the furthest one reached are holes an earlier session left —
    // the resume brief says never to go back for them — while the ones
    // AHEAD are simply where the module stopped. Naming the first hole as
    // "where to pick up" sent the next session back into it, and listing
    // the untaught remainder as holes described the whole rest of the
    // lesson as missing.
    const furthest = allCheckpoints.reduce(
      (acc: number, id: string, i: number) => (done.has(id) ? i : acc),
      -1,
    );
    const holes = missing.filter((id) => allCheckpoints.indexOf(id) < furthest);
    const ahead = missing.filter((id) => allCheckpoints.indexOf(id) > furthest);
    if (holes.length) {
      out.push(
        `## Not recorded`,
        `These checkpoints have no record — an earlier session ended before they were ` +
          `reached, and later ones carried on without them: ${holes.join(", ")}`,
        "",
      );
    }
    if (ahead.length) {
      out.push(`## Where to pick up`, `Next checkpoint: ${ahead[0]}`, "");
    }
  }
  return out.join("\n");
}

const JUDGMENTS = ["pass", "pass_with_hints", "guided", "prediction"];

const MARIMO_CELL_RULES =
  "Cell code rules (marimo is reactive): " +
  "(1) NEVER read a widget's .value in the cell that creates it — marimo forbids it. " +
  "Pattern: one cell makes and displays the widget (w = mo.ui.slider(…) then w as last line), " +
  "a SECOND cell uses w.value. " +
  "(2) Do NOT import mo/nx/np/plt/ig/sns/alt/pd — they already exist (redundant imports are " +
  "stripped); netviz(edges, highlight=[...]) is also predefined for themed D3 network drawings. " +
  "(3) Each public variable is owned by exactly ONE cell; prefix throwaway names with _ . " +
  "(4) The cell's LAST expression is what gets displayed; markdown via mo.md(r'''…'''). " +
  "(5) A matplotlib figure renders ONLY as the cell's last expression — NEVER interpolate a " +
  "figure into an mo.md f-string (it prints object gibberish, not an image). UI widgets may " +
  "be embedded in mo.md f-strings; figures may not. " +
  "(6) Text AND a figure in ONE cell: end with mo.vstack([mo.md(r'''…'''), <figure or " +
  "netviz(...)>]). NEVER draw a diagram as ASCII art inside markdown — a tiny netviz " +
  "(it even draws self-loops) or matplotlib figure always looks better.";

export default function (pi: ExtensionAPI) {
  piRef = pi;
  // The notebook server, the student's copy of the notebook, and the browser
  // page all come up here — nothing outside this package launches anything.
  // Not awaited: uv's first sandbox build can take a minute, and the student's
  // first turn is a hello. runKernel waits when waiting actually matters.
  void marimoUrl();
  // The tutor talks; it does not run commands. Every notebook and log
  // operation goes through the quiet nb_* tools, so a raw shell would only
  // ever scroll past the student mid-lesson.
  try {
    const active: string[] = pi.getActiveTools?.() ?? [];
    if (active.includes("bash")) pi.setActiveTools?.(active.filter((n) => n !== "bash"));
  } catch {
    /* an older pi without tool management: AGENTS.md still forbids it */
  }
  // Guards against a checkpoint's build landing in the wrong place: if the
  // tutor starts building checkpoint B before closing checkpoint A with
  // checkpoint_done, A's note cell gets created LATE and lands after B's
  // build cells instead of before them (seen in production — a "welcome"
  // note appeared after the next checkpoint's image). nb_add_template (and
  // nb_add_exercise, when tagged) check this before inserting.
  //
  // The open checkpoint is armed from the SCRIPT, not from the build: a
  // checkpoint with `build: none` (cp0_welcome, cp5_tension) never inserts a
  // cell, so arming on insert left exactly those unguarded — which is how
  // cp0's welcome note landed after cp1's image. Armed at session start and
  // advanced to the next scripted id by checkpoint_done.
  let pendingCheckpoint: string | null = null;
  // A slot-drift refusal fires at most once per checkpoint, so a model that
  // cannot satisfy it can never trap the student in a retry loop.
  const slotDriftWarned = new Map<string, number>();
  // Same idea for the souvenir contract: a text-only detour cell is bounced
  // once, then accepted, so a detour that genuinely has no picture in it can
  // still be recorded.
  // Keyed by the CELL, and counted. It was keyed by the question STRING, so
  // a retry that reworded the question was a fresh key: four bounces, four
  // rewordings, and the detour never reached the record at all.
  const detourTextOnlyWarned = new Map<string, number>();
  // Improvised-cell review refusals, per cell name — capped like the rest.
  const cellReviewWarned = new Map<string, number>();
  // chapter_done's "was this chapter actually taught?" refusals, per chapter.
  const chapterGateWarned = new Map<string, number>();
  // Checkpoints whose pace question was never actually put to the student
  // (the picker could not run). The next build for them is bounced once.
  const paceUnasked = new Set<string>();
  // The build-ordering refusal, per checkpoint. It was the last uncapped
  // guard in the file: whenever the open checkpoint ended up wrong, every
  // build refused forever and the only escape it named wrote a duplicate row.
  const buildOrderWarned = new Map<string, number>();
  // The late-close refusal, per checkpoint. A close that arrives long after
  // the answer did takes the NEXT checkpoint's turns with it — the capture
  // only resets here — and the model's own hint count has stopped matching
  // what happened. See the guard in checkpoint_done.
  const lateCloseWarned = new Map<string, number>();
  // Checkpoints an EARLIER session left unlogged. Reported to the tutor at
  // resume and then treated as settled: the brief says not to go back for
  // them, so nothing downstream may order it to.
  const resumeGaps = new Set<string>();
  // How many turns the tutor has spent on the checkpoint it is working now.
  // Reset by checkpoint_done and chapter_done — see the ⚖️ nudge at turn_end.
  let turnsInCheckpoint = 0;
  const STUCK_TURNS = 12;
  // Upload widgets nb_view_image has actually looked at. Cell presence is
  // not evidence that the photo was ASKED for: cp5's script builds the drop
  // area up front, so the area exists from question 1 and a tutor that then
  // skips the "photograph that page" step closes a paper checkpoint with no
  // paper in it — which is exactly what a live run did.
  const viewedPhotos = new Set<string>();

  pi.on("tool_result", async (event: any) => {
    recordPickedAnswer(event);
  });

  // ── The student's "Send to my tutor" button ───────────────────────────
  // marimo runs in its own process, so a button in the page cannot call a
  // tool. The upload templates append the widget's name to
  // session_artifacts/student_signal.txt instead, and this watcher turns
  // each new line into a turn. Without it the student has to type "it's up"
  // in the terminal — one more thing to explain, and one more place for a
  // nervous beginner to sit waiting for permission.
  //
  // The message starts with "The student clicked", which INJECTED_PREFIX
  // filters: a button press is the student acting, but it is not their
  // words, and it must never be logged as something they said.
  const signalPath = () =>
    path.join(process.cwd(), "session_artifacts", "student_signal.txt");
  const readSignalLines = (): string[] => {
    try {
      return fs
        .readFileSync(signalPath(), "utf-8")
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
    } catch {
      return [];
    }
  };
  // Whatever is already in the file belongs to an earlier session.
  let signalMark = readSignalLines().length;
  const signalTimer = setInterval(() => {
    try {
      const lines = readSignalLines();
      if (lines.length < signalMark) signalMark = lines.length; // reset/archived
      if (lines.length <= signalMark) return;
      const widget = lines[lines.length - 1];
      signalMark = lines.length;
      if (!/^[A-Za-z_]\w*$/.test(widget)) return;
      // The ⚖️ "Tutor gets stuck" button appeals over the tutor's head:
      // the referee model rules, then the verdict message starts the turn.
      if (widget === "tutor_stuck") {
        void handleAppeal(pi);
        return;
      }
      // Two kinds of box send: a photo drop area, and an exercise code box
      // (whose widget name ends in _ed). They need opposite instructions —
      // one is read with the vision model, the other with nb_read — and a
      // tutor told to call nb_view_image on a code box burns a turn on an
      // error the student then watches it recover from.
      const isCode = /_ed$/.test(widget);
      pi.sendMessage(
        {
          customType: "student-signal",
          content: isCode
            ? `The student clicked "Send my code to my tutor" in the notebook: they have ` +
              `run their code and are handing it in. Read it with ` +
              `nb_read(["${widget}.value"]) now — do not ask them to paste it. Their ` +
              `output is already on screen under the box; react to what THEIR code and ` +
              `chart actually show, and ask the checkpoint's question about it. If the ` +
              `code does not run or does not do the task, say so warmly and point at the ` +
              `one line to change — they can edit and press ▶ Run again as often as they ` +
              `like, then send again.`
            : `The student clicked "Send to my tutor" in the notebook: their photo is ` +
              `uploaded and showing in the "${widget}" box. Call ` +
              `nb_view_image(widget="${widget}", …) now — do not ask them whether it is ` +
              `up. Judge what you see against the task, and say something specific about ` +
              `THEIR picture. If it does not show what the checkpoint asked for, say so ` +
              `warmly and ask them to redo it and drop the new photo into the same box; ` +
              `they can replace it as many times as they need.`,
          display: false,
        },
        { deliverAs: "followUp", triggerTurn: true },
      );
    } catch {
      // the button is a convenience — never let it break a turn
    }
  }, 2000);
  (signalTimer as any).unref?.();

  // Normal exit, /resume, /new: take the notebook server with us. A marimo
  // left behind holds port 2718 against the next session, and its kernel still
  // has the previous student's variables in it.
  pi.on("session_shutdown", async () => {
    stopMarimo();
  });

  // Rendered plainly, in the warning colour, because it is the one message in
  // this whole toolkit written for the student to act on rather than read.
  pi.registerMessageRenderer("setup-help", (message: any, _opts: any, theme: any) => {
    const body = String(message.content ?? "")
      .split("\n")
      .map((l: string) => theme.fg("warning", l))
      .join("\n");
    return new Text(body, 0, 0);
  });

  pi.on("session_start", async (_event, _ctx) => {
    lastCtx = _ctx ?? lastCtx;
    // Before anything else: is the tutor's own model reachable? A broken key
    // or a stale endpoint is otherwise discovered as an unexplained
    // "Connection error." on the student's first hello.
    void preflightProvider(_ctx).then((problem) => {
      if (!problem) return;
      try {
        // No ui.notify beside this: pi renders a notification as another
        // "Error:" row, and the student is already looking at one they cannot
        // read. One block, in plain words, last on the screen.
        piRef?.sendMessage(
          {
            customType: "setup-help",
            content: `⚠  Your tutor cannot reach the course server.\n\n${problem}`,
            display: true,
          },
          { deliverAs: "followUp" },
        );
      } catch {
        /* best effort — the student still gets pi's own error */
      }
    });
    // ── Chapter start + resume brief ──────────────────────────────────────
    // Determine the current chapter (from progress or saved state), inject
    // its script, and — when previous progress exists — a resume brief that
    // asks the student continue-or-fresh.
    try {
      const chapters = loadChapters();
      if (chapters.length > 0) {
        const entries = readSessionLog();
        const cps = entries.filter((e: any) => e.type === "checkpoint" && e.id);
        // An empty log means the module has NOT started, whatever
        // chapter_state.json says — that file outlives the log it belongs
        // to. A fresh start archives the log; if the saved chapter were
        // still trusted after that, the "clean slate" session would open in
        // the middle of the module, on top of whatever cells the wipe left
        // behind. Seen exactly that way: no log, chapter_state "ch3", and a
        // notebook whose first cell was chapter 3's heading.
        let chapter =
          cps.length > 0
            ? (chapters.find((c) => c.id === currentChapterId()) ?? chapters[0])
            : chapters[0];
        pendingCheckpoint = chapter.checkpoints[0] ?? null;
        let moduleFinished = false;
        if (cps.length > 0) {
          const order = chapters.flatMap((c) => c.checkpoints);
          // Scan BACKWARDS for the last id the script actually knows. The
          // newest entry can be off-script — a stretch (cs1_code), a typo —
          // and indexOf(-1)+1 would silently resolve to cp0_welcome, walking
          // a student who finished four chapters back to the welcome.
          // The FURTHEST scripted checkpoint reached, not the newest row.
          // A practice round logged out of order (a `cp2_distance_extra`
          // after chapter 3) rewound the whole session and armed the build
          // guard back at chapter 2, while the closing summary still read
          // the position correctly — the two disagreed on the same log.
          const lastScripted = cps
            .map((e: any) => baseCheckpointId(String(e.id ?? "")))
            .filter((id: string) => order.includes(id))
            .reduce(
              (acc: string | undefined, id: string) =>
                acc === undefined || order.indexOf(id) > order.indexOf(acc) ? id : acc,
              undefined as string | undefined,
            );
          // Positional, matching nextCheckpointId — see the note there for
          // why resuming INTO a gap is not supported. No next id means the
          // module is done; do NOT fall back to the last checkpoint, which
          // tells the tutor to re-run cp8, whose re-close skips the existing
          // session_record while overwriting session_summary.md.
          const nextId = lastScripted ? order[order.indexOf(lastScripted) + 1] : order[0];
          const finished = !nextId;
          // A gap is reported, not repaired: the brief says so in one line so
          // the tutor does not silently carry on as if the module were whole,
          // and the closing summary lists the missing ids either way.
          const doneIds = new Set(cps.map((e: any) => baseCheckpointId(String(e.id))));
          const gaps = order.filter(
            (o: string) => !doneIds.has(o) && (!nextId || order.indexOf(o) < order.indexOf(nextId)),
          );
          resumeGaps.clear();
          for (const g of gaps) resumeGaps.add(g);
          moduleFinished = finished;
          chapter = (nextId && chapters.find((c) => c.checkpoints.includes(nextId))) || chapter;
          pendingCheckpoint = nextId ?? null;
          // The continue-or-fresh answer below is session mechanics, not a
          // lesson answer, and nothing drains the pick buffer until the next
          // checkpoint_done — so without this it rode into that checkpoint's
          // record and printed as *You chose: "Continue where we left off"*
          // under a lecture question in the submitted notebook.
          awaitingResumeChoice = true;
          pi.sendMessage(
            {
              customType: "resume-brief",
              content:
                `RESUME CONTEXT (invisible to the student — never mention this message): ` +
                `a previous session exists. Progress so far:\n${progressBrief(entries)}\n` +
                (gaps.length
                  ? `Note for you, not for them: ${gaps.join(", ")} ${gaps.length > 1 ? "have" : "has"} ` +
                    `no record — an earlier session ended between them. Do NOT go back for ` +
                    `${gaps.length > 1 ? "them" : "it"}; carry on from where the module is now, and the ` +
                    `closing summary will show ${gaps.length > 1 ? "them" : "it"} as missing.\n`
                  : "") +
                `FIRST, greet the student and ask with ask_user_question, using EXACTLY these ` +
                `two option labels: "Continue where we left off" and "Start fresh" — the ` +
                `extension recognises those words and keeps this housekeeping answer out of ` +
                `the graded record, so do not reword them. If they choose fresh: call ` +
                `nb_fresh_start and follow its instructions (chapter 1 reloads ` +
                `automatically — do not improvise). ` +
                (finished
                  ? (fs.existsSync(path.join(process.cwd(), "session_artifacts", "session_summary.md"))
                      ? `If they continue: they already FINISHED this module — do not re-run any ` +
                        `checkpoint and do not call chapter_done. Say so warmly, offer to answer ` +
                        `questions or replay any experiment in the notebook, and log anything you ` +
                        `answer with log_detour.`
                      // Every checkpoint is logged but the closing record was never
                      // written — the last session ended between the final answer and
                      // chapter_done. Seen live: the tutor said a warm goodbye and the
                      // student's notebook shipped with no session_record and no
                      // summary, the two things the grader opens first.
                      : `If they continue: every checkpoint is logged but this module was ` +
                        `never CLOSED — there is no session_record cell and no summary. Do ` +
                        `not re-run any checkpoint. Say one warm line that you are just ` +
                        `filing their work, call chapter_done to write the closing record, ` +
                        `and then say goodbye.`)
                  : `If they continue: do NOT rebuild existing notebook cells ` +
                    `(nb_add_template skips duplicates automatically), remind them in one ` +
                    `sentence where you two left off, and continue at checkpoint ${nextId} ` +
                    `(chapter "${chapter.title}").`),
              display: false,
            },
            { deliverAs: "nextTurn" },
          );
        }
        writeChapterState(chapter.id);
        const num = chapters.findIndex((c) => c.id === chapter.id) + 1;
        // A finished module gets NO chapter script: it ends "work its
        // checkpoints in order... then call chapter_done", which is the exact
        // opposite of the resume brief above, and re-running cp8 appends a
        // duplicate log row against a note cell that skips as already there.
        if (!moduleFinished) {
          pi.sendMessage(
            {
              customType: "chapter-script",
              content: chapterScriptMessage(chapter, num, chapters.length),
              display: false,
            },
            { deliverAs: "nextTurn" },
          );
        }
        // Delayed and retried: the kernel is still booting at session start,
        // and stays cold until a browser client connects. Each attempt
        // re-checks that this chapter is still current — if the student
        // chose "start fresh" (or a chapter transition otherwise happened),
        // inserting its header would land a stray "Chapter N" heading in
        // the middle of a different chapter's cells (seen in production).
        scheduleChapterHeader(chapter, num, chapters.length);
      }
    } catch {
      // chapter injection is best-effort; AGENTS.md tells the tutor how to cope
    }
  });

  // Chapter dividers render as a single accent line in the transcript.
  pi.registerMessageRenderer("chapter-divider", (message: any, _opts: any, theme: any) => {
    return new Text(theme.fg("accent", String(message.content ?? "")), 0, 0);
  });

  // Custom compaction at chapter boundaries: the handoff brief IS the summary
  // (deterministic, no extra LLM call).
  let pendingHandoffBrief: string | null = null;
  // When it was armed. The brief belongs to ONE compaction — the one
  // chapter_done starts — and every other path that arms it also clears it on
  // failure, except the 20-second fallback timer, which did not. A brief left
  // behind there is consumed by the next compaction instead, which may be the
  // automatic one forty minutes later: the tutor is then handed a summary
  // saying it has just started chapter 2, in the middle of chapter 3, and the
  // real conversation is gone. Stale after two minutes, which is far longer
  // than the handoff takes and far shorter than the session.
  let handoffArmedAt = 0;
  const HANDOFF_TTL_MS = 120_000;
  // Armed by chapter_done, fired at message_end — see the comment there.
  let pendingCompaction: (() => void | Promise<void>) | null = null;
  const runPendingCompaction = () => {
    const go = pendingCompaction;
    pendingCompaction = null;
    if (go) void Promise.resolve(go()).catch(() => {});
  };
  pi.on("session_before_compact", async (event: any) => {
    if (!pendingHandoffBrief) return;
    if (Date.now() - handoffArmedAt > HANDOFF_TTL_MS) {
      pendingHandoffBrief = null;
      return; // let pi summarise this one itself
    }
    const summary = pendingHandoffBrief;
    pendingHandoffBrief = null;
    return {
      compaction: {
        summary,
        firstKeptEntryId: event.preparation.firstKeptEntryId,
        tokensBefore: event.preparation.tokensBefore,
      },
    };
  });

  // ── chapter_done ──────────────────────────────────────────────────────────
  pi.registerTool({
    name: "chapter_done",
    label: "Chapter done",
    description:
      "Call when the current chapter's FINAL checkpoint has been logged. Pass short handoff " +
      "notes for the next part of the lesson (student's style, anchors worth reusing like " +
      "'their cable was the long one', anything to watch). The next chapter script loads " +
      "automatically — after calling, say ONE short bridge sentence and wait.",
    promptSnippet: "Finish the current chapter and load the next (with handoff notes)",
    parameters: Type.Object({
      status: STATUS_PARAM,
      handoff: Type.String({
        description: "2-4 sentences: student profile updates, anchors, watch-outs.",
      }),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx: any) {
      lastCtx = ctx ?? lastCtx;
      const chapters = loadChapters();
      const curId = currentChapterId() ?? chapters[0]?.id;
      const idx = chapters.findIndex((c) => c.id === curId);
      const next = chapters[idx + 1];

      // ── The chapter must actually have been taught ──────────────────────
      // A chapter transition compacts the conversation, and a model that
      // comes out of compaction disoriented can call chapter_done a second
      // time. That happened in a live session: two transitions fired back to
      // back and the student went from chapter 1 straight into chapter 3's
      // pen-and-paper task, having never met distance or clustering. The
      // student picker below cannot catch it — they were asked "anything
      // first?" about a chapter that had not happened.
      const cur = chapters[idx];
      if (cur) {
        const loggedIds = new Set(
          readSessionLog()
            .filter((e: any) => e?.type === "checkpoint" && e.id)
            .map((e: any) => baseCheckpointId(String(e.id))),
        );
        // Checkpoints an earlier session already skipped are NOT this
        // chapter's unfinished business: the resume brief told the tutor not
        // to go back for them, and this gate was then ordering it straight
        // back. Same rule both places — a gap is reported, not repaired.
        const missing = cur.checkpoints.filter(
          (c) => !loggedIds.has(c) && !resumeGaps.has(c),
        );
        // Capped like every other guard. Without a cap, a log that cannot be
        // written makes this fire forever — readSessionLog() returns [], so
        // every checkpoint reads as missing — while checkpoint_done is
        // telling the tutor "LOG WRITE FAILED — keep teaching". The student
        // could never leave chapter 1.
        const gateStrikes = chapterGateWarned.get(cur.id) ?? 0;
        // A referee ruling that ends the chapter walks through this gate;
        // the summary still reports the unlogged checkpoints as gaps.
        const chapterWaived = refereeWaiverActive();
        if (missing.length > 0 && chapterWaived) refereeWaiver = null;
        if (missing.length > 0 && gateStrikes < 2 && !chapterWaived) {
          chapterGateWarned.set(cur.id, gateStrikes + 1);
          return {
            content: [
              {
                type: "text" as const,
                text:
                  `NOT ADVANCED — "${cur.title}" is not finished. These checkpoints have ` +
                  `never been logged: ${missing.join(", ")}. Say nothing about this to the ` +
                  `student; just carry on with "${missing[0]}" from your CHAPTER SCRIPT, ` +
                  `ending it with checkpoint_done, and call chapter_done again once the ` +
                  `chapter's last checkpoint is logged.`,
              },
            ],
            details: { gated: true },
          };
        }
      }

      // ── Forced chapter-end follow-up ────────────────────────────────────
      // Enforced here, not by prompt: a chapter boundary is the one moment
      // the student must get an unhurried "anything first?" — the tutor was
      // racing past questions and requests for extra practice.
      const READY = next ? "I'm ready for the next chapter" : "I'm ready to wrap up";
      const ASK_Q = "I have a question first";
      const MORE = "Give me one more practice problem";
      if (ctx?.ui?.select) {
        const title = chapters[idx]?.title ?? "this part";
        const { choice, typed } = await askStudent(
          ctx,
          `Before we leave "${title}" — anything first?`,
          [READY, ASK_Q, MORE],
        );
        if (choice !== READY) {
          const text =
            choice === ASK_Q
              ? `The student has a QUESTION. Do NOT advance. Ask them in plain text what it ` +
                `is, answer it properly, leave a souvenir cell (mo.vstack: note + netviz/figure ` +
                `— never ASCII art), log the detour, then call chapter_done again.`
              : choice === MORE
                ? // Named objects belong to whichever module wrote them, and this
                  // string is read by every module that installs the toolkit: it
                  // used to send the Königsberg tutor off to improvise on "the
                  // 4-person network, the 8-dot ring", which are the small-world
                  // module's and exist nowhere in this one. The scripts already
                  // carry the right data for exactly this, per checkpoint.
                  `The student wants MORE PRACTICE. Do NOT advance. Improvise ONE problem of ` +
                  `the same kind on NEW data — the fresh_variants of the checkpoint you just ` +
                  `finished, or the same recipe on this module's own objects. Guide, judge, log ` +
                  `it as extra practice (never a fail), then call chapter_done again.`
                : choice === TYPE_IT && typed
                  ? `Do NOT advance. The student typed this instead of picking a row — these ` +
                    `are THEIR words, react to them: "${typed}". Answer whatever it asks (a ` +
                    `question gets a proper answer, a souvenir cell and log_detour), then ask ` +
                    `in plain text whether they are ready, and call chapter_done again once ` +
                    `they say so.`
                  : `The student closed the picker without choosing — they may want to say ` +
                    `something in their own words. Ask them in plain text what they'd like to do, ` +
                    `handle it, then call chapter_done again.`;
          return { content: [{ type: "text" as const, text }], details: { gated: true } };
        }
      } else if ((chapterGateWarned.get(`${chapters[idx]?.id}:pace`) ?? 0) < 1) {
        // No picker — a dead notebook, a headless restart. DESIGN.md says
        // chapter_done "refuses to transition" without the student's word,
        // and with no else branch it simply transitioned. checkpoint_done's
        // gate was fixed the same way; this is its twin.
        //
        // ONE refusal, because the picker will still be missing on the
        // retry: the point is to make the tutor ask, not to trap it in a
        // loop it can never satisfy.
        chapterGateWarned.set(`${chapters[idx]?.id}:pace`, 1);
        return {
          content: [
            {
              type: "text" as const,
              text:
                `NOT ADVANCED — the picker could not run, so the student has not been asked ` +
                `whether they want anything before leaving "${chapters[idx]?.title ?? "this part"}". ` +
                `Ask them in plain text — ready to move on, a question first, or one more ` +
                `practice problem — END YOUR TURN, and wait. Call chapter_done again once ` +
                `they answer; handle a question or a practice round first if that is what ` +
                `they choose.`,
            },
          ],
          details: { gated: true },
        };
      }

      if (!next) {
        // The closing artifacts are DERIVED from the log — never retyped from
        // the model's memory of the session (that is the graded record).
        const entries = readSessionLog();
        const allCps = chapters.flatMap((c) => c.checkpoints);
        let done = "";
        // Last chance for any note the notebook was down for — chapter 5 has
        // no builds at all, so without this a note parked at cp7 or cp8 would
        // never reach the page it belongs on.
        await flushParkedNotes(_signal);
        const rec = await insertMarkdownCell("session_record", buildSessionRecord(entries), _signal);
        done += rec.failed ? "session_record cell FAILED. " : "Closing record added to their notebook. ";
        try {
          fs.writeFileSync(
            path.join(process.cwd(), "session_artifacts", "session_summary.md"),
            buildSessionSummary(entries, allCps),
          );
          done += "Summary written. ";
        } catch {
          done += "Summary write failed. ";
        }
        return {
          content: [
            {
              type: "text" as const,
              text:
                `That was the FINAL chapter. ${done}\n` +
                // The last thing a finished session must do is hand itself in.
                // This used to end at "say goodbye" and nothing anywhere told
                // the student to submit — a whole module could be finished,
                // closed and left sitting in a folder, ungraded.
                `NOW CALL nb_submit — that hands their work in, and it is the last thing ` +
                `this session owes them. Then say goodbye: tell them plainly what they ` +
                `can now do, that their answers (not code) are what gets reviewed, that ` +
                `their work is handed in, and that the notebook is theirs to keep and ` +
                `keep playing with.`,
            },
          ],
          details: {},
        };
      }
      turnsInCheckpoint = 0;
      writeChapterState(next.id);
      // Re-arm the build-ordering guard on the new chapter's first
      // checkpoint, so it keeps working across the transition instead of
      // still pointing at the chapter that just ended.
      pendingCheckpoint = next.checkpoints[0] ?? pendingCheckpoint;
      const brief =
        `=== TUTORING HANDOFF (chapter transition, invisible to the student) ===\n` +
        `You are the SAME tutor, mid-session. Conversation so far, summarized:\n` +
        `Progress:\n${progressBrief(readSessionLog())}\n` +
        `Tutor's notes: ${params.handoff}\n` +
        `Anything above about a camera, a phone or a photo belongs to the checkpoint it ` +
        `happened at — it is NOT a standing fact about this student. Every paper ` +
        `checkpoint in this chapter asks for the photo, in one line, and waits.\n` +
        `The notebook already contains every cell built so far — never rebuild them. ` +
        `Continue warmly with the same voice; your new CHAPTER SCRIPT message has the curriculum.`;
      pendingHandoffBrief = brief;
      handoffArmedAt = Date.now();
      // The next chapter must load AFTER compaction: injecting before it
      // races the session reload (the fresh turn gets aborted and nothing
      // restarts — seen in production) and the script could be summarized
      // away. loadOnce also serves as the fallback when compaction errors
      // (e.g. nothing to compact) or never calls back.
      const num = idx + 2;
      let loaded = false;
      const loadOnce = () => {
        if (loaded) return;
        loaded = true;
        // Any note the notebook was down for goes in BEFORE the next
        // chapter's heading. A live run killed marimo during cp1_routing;
        // its note was parked, the chapter turned over, and the recovered
        // note landed under "── Chapter 2 ──" — a chapter-1 answer filed
        // inside chapter 2, in the keepsake the student submits. Every
        // other insertion point already flushes first; this was the one
        // that did not.
        void flushParkedNotes().then(() => insertChapterHeader(next, num, chapters.length));
        pi.sendMessage(
          {
            customType: "chapter-divider",
            content: `── Chapter ${num} · ${next.title} ──`,
            display: true,
          },
          { deliverAs: "followUp" },
        );
        pi.sendMessage(
          {
            customType: "chapter-script",
            content: chapterScriptMessage(next, num, chapters.length),
            display: false,
          },
          { deliverAs: "followUp", triggerTurn: true },
        );
      };
      // Compaction aborts whatever model request is in flight — and the one
      // in flight right now is the tutor saying the bridge sentence this very
      // tool result asks for. Started here, it printed a red
      // "Error: This operation was aborted" into the student's terminal at
      // every chapter turn, four times a session, with nothing actually
      // wrong. So it is armed here and fired at message_end, once the tutor
      // has finished speaking.
      pendingCompaction = async () => {
        // Wait for pi to actually settle. turn_end fires while the run is
        // still winding down, and compacting then aborts it — which the
        // student reads as "Error: This operation was aborted", in red, at
        // every chapter boundary. isIdle() is the difference between "the
        // tutor has stopped talking" and "there is nothing left to abort".
        for (let i = 0; i < 40 && ctx?.isIdle && !ctx.isIdle(); i++) {
          await new Promise((r) => setTimeout(r, 250));
        }
        try {
          ctx?.compact?.({
            customInstructions: "chapter handoff",
            onComplete: loadOnce,
            onError: () => {
              pendingHandoffBrief = null;
              loadOnce();
            },
          });
          const timer = setTimeout(loadOnce, 20_000);
          (timer as any).unref?.();
        } catch {
          pendingHandoffBrief = null;
          loadOnce();
        }
      };
      // A floor under it: a tutor that says nothing at all may never produce
      // a message_end, and a chapter that never loads strands the student.
      const armed = setTimeout(runPendingCompaction, 30_000);
      (armed as any).unref?.();
      return {
        content: [
          {
            type: "text" as const,
            text:
              `Handoff recorded. Say ONE short, warm bridge sentence to the student ` +
              `(no new questions) and END YOUR TURN — chapter "${next.title}" loads automatically.`,
          },
        ],
        details: {},
      };
    },
    ...quiet("Getting our next chapter ready…"),
  });

  // ── Runaway guard ─────────────────────────────────────────────────────────
  // Silent safety net only (style is steered by AGENTS.md, not enforced):
  // flash-class models can fall into degenerate repetition loops. Abort the
  // generation if a single message runs absurdly long, and nudge a restart.
  const RUNAWAY_CHARS = 1600;
  let runawayFired = false;
  pi.on("message_update", async (event: any, ctx: any) => {
    const msg = event?.message;
    if (msg?.role !== "assistant" || runawayFired) return;
    const raw = msg.content;
    const t =
      typeof raw === "string"
        ? raw
        : (Array.isArray(raw) ? raw : [])
            .filter((c: any) => c?.type === "text")
            .map((c: any) => c.text ?? "")
            .join("\n");
    if (t.length <= RUNAWAY_CHARS) return;
    runawayFired = true;
    try {
      ctx.abort();
    } catch {
      // best-effort
    }
    pi.sendMessage(
      {
        customType: "runaway-guard",
        content:
          "NOTE (invisible to the student): your message ran away and was cut off. " +
          "Continue with one short message.",
        display: false,
      },
      { deliverAs: "followUp", triggerTurn: true },
    );
  });
  pi.on("message_end", async () => {
    runawayFired = false;
  });

  // Fixed-choice questions go through the ask_user_question tool
  // (@juicesharp/rpiv-ask-user-question package, declared in .pi/settings.json).

  // ── checkpoint_done ───────────────────────────────────────────────────────
  // One call replaces the whole per-checkpoint ceremony: the extension writes
  // the graded log (with the student's own messages captured from the
  // transcript), renders the note cell from the chapter script's `note:`
  // skeleton, and runs the transition ask itself. The tutor supplies only
  // what a model can: the verbatim answer and the judgment.
  pi.registerTool({
    name: "checkpoint_done",
    label: "Checkpoint done",
    description:
      "Finish a checkpoint: this ONE call logs it (graded artifact), adds the notebook note " +
      "cell from the chapter script's note: skeleton, and asks the student whether to move " +
      "on. Call it right after you judge their answer — never hand-write log JSON, never " +
      "hand-write the note cell. The result tells you what the student chose: only 'ready' " +
      "means you may start the next checkpoint.",
    promptSnippet: "Log a checkpoint, add its note cell, and ask the student what's next",
    promptGuidelines: [
      "End EVERY checkpoint with checkpoint_done — it replaces hand-written log JSON, the note cell, and the transition question.",
    ],
    parameters: Type.Object({
      status: STATUS_PARAM,
      id: Type.String({ description: "Checkpoint id from the script, e.g. 'cp2_distance'." }),
      // "As you asked it", not "as the script wrote it". A student who
      // answers a later question early is answering one you never put, and a
      // row that lists both reads back as a tutor firing two questions in one
      // breath — which is what a reviewer concluded from exactly this row,
      // from the log alone, with the transcript already gone.
      question: Type.String({
        description:
          "The question as you actually asked it — not the script's wording, and " +
          "not a question you skipped because they answered it early. Say in notes " +
          "when they volunteered the rest.",
      }),
      student_response: Type.String({
        description: "Their answer VERBATIM — their words, not your summary.",
      }),
      judgment: Type.String({
        description: "One of: pass | pass_with_hints | guided | prediction.",
      }),
      // Optional, and accepted as a string too: execute already does
      // `Number(params.hints_used ?? 0) || 0`, so a missing or "2"-shaped
      // value costs nothing — while requiring a number meant a model that
      // sent "2" lost the entire close, including the student's verbatim
      // answer, to a schema rejection it never sees the text of.
      hints_used: Type.Optional(
        Type.Union([Type.Number(), Type.String()], {
          description:
            "How many hints you gave. A hint is any question you asked BECAUSE " +
            "their answer was not there yet — one counts, even if they got it on " +
            "the next turn. 0 only if they answered every part first time.",
        }),
      ),
      notes: Type.String({ description: "One line: what their answer showed." }),
      // A union with String because a small model reliably sends this array
      // as a JSON string sooner or later, and a schema rejection here loses
      // the whole close — the log row and the student's own words with it.
      // Normalised in execute; the shape it asks for is still the array.
      note_slots: Type.Optional(
        Type.Union([Type.Array(Type.String()), Type.String()], {
          description:
            "Fills for the «slots» in the script's note: skeleton that a transcript " +
            "cannot supply — what their drawing shows, which option they picked, the " +
            "numbers a widget displayed. ONE per such slot, in order, as an array of " +
            "strings. Slots marked «… verbatim» are filled with the student's typed " +
            "words automatically — skip them. Sending fewer than the rest is refused " +
            "twice, then the unfilled ones print as '(not answered)' in the graded " +
            "notebook.",
        }),
      ),
      note_markdown: Type.Optional(
        Type.String({
          description:
            "Only when the script has no note: skeleton — the full note cell markdown " +
            "(plain-words title, 2-4 sentences with $math$, then their quoted answer).",
        }),
      ),
    }),
    async execute(_id, params, signal, _onUpdate, ctx: any) {
      lastCtx = ctx ?? lastCtx;
      // Everything below looks the checkpoint up by this id — the note
      // skeleton, the build guard, the ordering guard, both closing artifacts.
      // A near miss is pulled back onto the script rather than quietly
      // disabling all four (see snapCheckpointId).
      const snapped = snapCheckpointId(String(params.id ?? "").trim());
      const id = snapped.id;
      const judgment = String(params.judgment ?? "").trim();
      if (!JUDGMENTS.includes(judgment)) {
        return toResult({
          out: `NOT LOGGED — judgment must be one of ${JUDGMENTS.join(" | ")}, got "${judgment}". Call again.`,
          failed: false,
        });
      }
      let response = String(params.student_response ?? "").trim();
      if (!response) {
        return toResult({
          out: `NOT LOGGED — student_response is empty. Log their actual words (or "(no answer — moved on)") and call again.`,
          failed: false,
        });
      }
      // Their answer to a question that is still hanging would be eaten by
      // this tool's own dialog. Once — a rhetorical closer is one retry away,
      // and the retry needs no speech.
      const hanging = tutorAwaitingAnswer(ctx);
      if (hanging && (slotDriftWarned.get(`${id}:hanging`) ?? 0) < 1) {
        slotDriftWarned.set(`${id}:hanging`, 1);
        return toResult({
          out:
            `NOT LOGGED — you just asked "${hanging.slice(0, 90)}" and they have not ` +
            `answered. Closing here opens the "where to next?" dialog, which takes over ` +
            `the keyboard: their answer would go into the picker and never reach you, ` +
            `and the checkpoint would close itself. Say nothing more, WAIT for their ` +
            `reply, react to it, and then call checkpoint_done.\n` +
            `If that question was rhetorical (a cliffhanger into the next chapter) or ` +
            `they have already answered it, call checkpoint_done again right now — no ` +
            `need to say anything first.`,
          failed: false,
        });
      }
      // A checkpoint whose script promises a build cannot be closed while
      // that build is missing. cp5_ring_formula asks seven things and the
      // seventh is "photograph that page"; a live run answered the first
      // six, decided the reasoning was complete, and closed — the upload
      // area was never inserted and the photo never asked for, on a
      // checkpoint whose whole point is the page in the student's own hand.
      // Capped like the rest, so a dead notebook cannot strand anyone.
      const wantCells = scriptedBuildCells(baseCheckpointId(id));
      let missingBuild: string[] = [];
      if (wantCells.length > 0) {
        const have = await notebookCellNames(signal);
        if (have) missingBuild = wantCells.filter((c) => !have.includes(c));
      }
      const buildStrikes = slotDriftWarned.get(`${id}:build`) ?? 0;
      if (missingBuild.length > 0 && buildStrikes < 2 && !refereeWaiverActive()) {
        slotDriftWarned.set(`${id}:build`, buildStrikes + 1);
        return toResult({
          out:
            `NOT LOGGED — this checkpoint's build never happened: ${missingBuild
              .map((c) => `"${c}"`)
              .join(", ")} ${missingBuild.length > 1 ? "are" : "is"} missing from the ` +
            `notebook. Read your script's build: line for "${baseCheckpointId(id)}", run ` +
            `the nb_add_template it names, and work the part of the ask that goes with it ` +
            `— on a paper checkpoint that means asking for the photo and WAITING for the ` +
            `📨 Send press. Then call checkpoint_done again.`,
          failed: false,
        });
      }
      // The build existing is not the same as the page being asked for. On a
      // checkpoint whose build inserts a drop area, either a photo came
      // through (nb_view_image saw it) or the student said they cannot take
      // one — and in that case the record should say so. One nudge, then it
      // logs either way with the gap on the row.
      const wantPhotos = scriptedPhotoCells(baseCheckpointId(id));
      // The in-memory set dies with the process, but the photo itself does
      // not: nb_view_image saves the original under assets/uploads/. Consult
      // both, or a resume after a photo was already read re-fires the
      // refusal and stamps a false `photo_missing` on the graded row.
      const uploadedOnDisk = (w: string): boolean => {
        try {
          return fs
            .readdirSync(path.join(process.cwd(), "assets", "uploads"))
            .some((f) => f.startsWith(`${w}_upload`) || f.startsWith(`${w}_view`));
        } catch {
          return false;
        }
      };
      const photoMissing =
        wantPhotos.length > 0 &&
        !wantPhotos.some((w) => viewedPhotos.has(w) || uploadedOnDisk(w));
      // Did they say ANYTHING about a camera at THIS checkpoint? A live run
      // carried "camera's broken" forward from cp2_paperwork and opened cp4
      // with "well, thinking time, since your camera's still out" — the page
      // never asked for at all, on a checkpoint whose script says "THE PAGE
      // IS THE POINT". So the exemption is scoped to where it was said: they
      // may repeat it in one word, but the ask has to happen.
      // Has the ask been put to them here? Either they mentioned a camera in
      // their own words, or — after the first refusal told the tutor to ask —
      // they have typed something since. A live run refused three times
      // because the student answered "still broken sorry, ill go with the
      // typed version", which says nothing about a camera; they had answered
      // the question, and the guard did not notice.
      const photoStrikes = slotDriftWarned.get(`${id}:photo`) ?? 0;
      const saidSoFar = studentSaidSince(ctx, false);
      const cameraSaidHere =
        saidSoFar.some((m) => /camera|photo|picture|scan|phone/i.test(m)) ||
        (photoStrikes > 0 && saidSoFar.length > 0);
      if (photoMissing && photoStrikes < (cameraSaidHere ? 1 : 2) && !refereeWaiverActive()) {
        slotDriftWarned.set(`${id}:photo`, photoStrikes + 1);
        return toResult({
          out:
            `NOT LOGGED — this is a pen-and-paper checkpoint and no photo has reached ` +
            `me for "${wantPhotos[0]}". If you have not asked yet: SAY THE ASK OUT LOUD ` +
            `to the student now — the page and nothing else — and only then end your ` +
            `turn. Never end a turn on this without speaking: a silent turn is a frozen ` +
            `screen, and in a live run it left a student waiting ten minutes. Their 📨 ` +
            `Send press starts the next turn, and you read it with nb_view_image.\n` +
            (cameraSaidHere
              ? `They have said something about a camera here, so if that was "I can't", ` +
                `their typed work counts — say so in notes ("camera broken, typed ` +
                `instead") and call checkpoint_done again; it will log.`
              : `They have said NOTHING about a camera at this checkpoint. An answer they ` +
                `gave two checkpoints ago is not an answer here: ask for the page, and if ` +
                `they tell you again that they cannot, that is fine and their typed work ` +
                `counts.`),
          failed: false,
        });
      }
      // Peek, don't consume: a refusal below must leave the transcript mark
      // where it was, or the retry would log an empty student_said_verbatim.
      const said = studentSaidSince(ctx, false);
      const picked = pickedSince(false);
      // Note «slots» the instructor marked «… verbatim» are held to the
      // student's own words. «their pick» (a picker choice, which never
      // reaches the transcript) and free commentary slots are the tutor's to
      // phrase, and are not checked. The bar is deliberately low — reordering
      // and joining and labelling are all fine — so that only invention
      // trips it: a figure they never gave, or three added content words.
      //
      // The pool is the transcript plus the QUESTION's own vocabulary —
      // echoing the words of the question ("distance", "average") is
      // labelling, not fabrication. student_response is deliberately NOT in
      // it: it is model-authored, so including it let the model whitelist
      // its own paraphrase by writing the same text into both fields, which
      // is exactly the failure the check exists to catch.
      //
      // With nothing typed there is nothing to check against, so the check
      // stands down rather than guessing. Checkpoints answered by drawing or
      // photograph are marked in the script instead: their slots do not say
      // «verbatim», because the tutor legitimately writes what the picture
      // shows there, and holding that to the transcript refused honest
      // records (it did, for every photo checkpoint).
      // `picked` belongs in the pool as much as `said` does: a dialog choice
      // IS the student's own answer. Leaving it out refused cp1's honest
      // record twice ("about 20" — the figure 20 appears in neither the
      // typed follow-up nor the question) and then stamped a false quoting
      // warning on it in the submitted notebook.
      const pool = [...said, ...picked, String(params.question ?? "")];
      const markers = slotMarkers(noteSkeleton(id));
      // A «verbatim» slot is filled from the transcript, by the extension.
      // Asking the model to pair answers with labelled slots failed five
      // different ways across five rounds — dropped halves, a fragment
      // instead of the answer, quotes shifted by one, punctuation tidied —
      // and every deterministic guard that tried to catch it had to be
      // withdrawn for refusing honest records. The words are right here.
      // Their ANSWER, not the whole stream: asides they already got a
      // souvenir for, and pure acknowledgement, are dropped — unless the
      // message IS the answer the tutor logged, which rescues a one-word
      // "right" that happens to look like filler.
      const quotedFrom: number[] = [];
      const answerish = said.filter((m, i) => {
        const isResponse =
          normMsg(m) === normMsg(response) || bigramDice(m, response) >= 0.6;
        const keep = isResponse || (!detourAsked.has(normMsg(m)) && !isFillerMessage(m));
        if (keep) quotedFrom.push(i + 1);
        return keep;
      });
      const verbatimFill = answerish.length
        ? answerish.map((m) => `"${m.replace(/\n+/g, " ").trim()}"`).join(" · ")
        : response;
      // The model is asked for the OTHER slots only — the ones whose answer
      // was a drawing, a photo or a picker, which no transcript holds. It may
      // still send a fill per slot positionally (older habit, and what the
      // refusal below prints); both shapes are accepted.
      // Accepted as an array, or as the JSON string a small model sends when
      // it forgets the difference, or as one plain fill for a one-slot note.
      const rawSlots = params.note_slots;
      const slotList: unknown[] = Array.isArray(rawSlots)
        ? rawSlots
        : typeof rawSlots === "string" && rawSlots.trim()
          ? (() => {
              try {
                const parsed = JSON.parse(rawSlots);
                return Array.isArray(parsed) ? parsed : [rawSlots];
              } catch {
                return [rawSlots];
              }
            })()
          : [];
      const givenSlots = slotList.map((x: unknown) => String(x ?? ""));
      const modelSlotIdx = markers
        .map((m, i) => (/verbatim/i.test(m) ? -1 : i))
        .filter((i) => i >= 0);
      // Positional ONLY when the count matches the skeleton exactly. Anything
      // else is read as compact — three entries on a two-marker skeleton used
      // to make `modelFill(1)` read entry 2 while the model's real fill sat
      // unread at entry 1, and the note rendered the wrong one silently.
      const compactFills =
        modelSlotIdx.length < markers.length && givenSlots.length !== markers.length;
      const modelFill = (i: number) =>
        compactFills ? (givenSlots[modelSlotIdx.indexOf(i)] ?? "") : (givenSlots[i] ?? "");
      // A model-filled slot describes a picture, so its prose is the tutor's.
      // Anything it puts in QUOTATION MARKS is still the student's, and a
      // live run put three different spellings of one sentence into the
      // record — student_response, the transcript capture, and a note cell
      // reading "becuase tirangles are im[portat]", brackets and all. A quote
      // that is a near-copy of something they typed is replaced with what
      // they actually typed; a quote of three words or more that matches
      // nothing they said is refused, like any other invented content.
      const quotesSnapped: string[] = [];
      const quotesInvented: string[] = [];
      const fixQuotes = (fill: string): string =>
        fill.replace(/[\u201C"]([^\u201D"]{4,})[\u201D"]/g, (whole, inner: string) => {
          const text = inner.trim();
          let best = "";
          let score = 0;
          // The SAME pool the sibling drift check uses. Comparing against
          // `said` alone accused a student of inventing their own picker
          // answer — a choice never reaches the transcript — and that is the
          // second time this file has learned it: two refusals mid-lesson,
          // then a false "⚠ Quoting check" on the notebook they submit.
          for (const msg of pool) {
            const d = bigramDice(text, msg);
            if (d > score) {
              score = d;
              best = msg;
            }
          }
          if (score >= 0.8 && best && normMsg(best) !== normMsg(text)) {
            quotesSnapped.push(text);
            return `\u201C${best}\u201D`;
          }
          // Short quotes are labels and readings ("2.07", "long"), not speech.
          if (score < 0.5 && text.split(/\s+/).filter(Boolean).length >= 3) {
            quotesInvented.push(text);
          }
          return whole;
        });
      const filledSlots = markers.map((m, i) =>
        /verbatim/i.test(m) ? verbatimFill : fixQuotes(modelFill(i)),
      );
      const problems: string[] = [];
      // Stand down on the STUDENT-derived half only. `pool` also carries the
      // tutor's own question, which is never empty, so gating on it made this
      // check fire on a photo checkpoint where the student had typed nothing
      // — two refusals and a false "⚠ Quoting check" on the submitted
      // notebook, for a slot whose whole job is describing a picture.
      if ((said.length > 0 || picked.length > 0) && quotesInvented.length > 0) {
        problems.push(
          `these are in quotation marks in a note slot but the student never said them: ` +
            quotesInvented.map((q) => `"${q}"`).join(", "),
        );
      }
      // One fill per slot. The skeletons ask for a slot per part of the ask
      // precisely so the keepsake quotes the ANSWER and not whichever
      // fragment came last — under-filling turns that back into one sentence
      // repeated under three labels.
      // Numbers the student typed that reach no slot. RECORDED, not
      // enforced — for the fourth time in this file, a guard that tried to
      // decide which words belong in which slot had to be withdrawn.
      // `said` holds every message since the last checkpoint (detour
      // questions included, because log_detour peeks without consuming), so
      // "they typed 6/7 and no slot quotes it" fires on a student who
      // self-corrected to 7/6, on an aside asking "was that 1967?", on a
      // follow-up echoing a number the reveal just gave them, and on a
      // tutor who quoted 1.17 where they wrote 7/6. And the refusal told
      // the tutor to put those into the graded note. So it goes in the log
      // beside slot_sources, where a person can weigh it.
      // Only where a slot could have quoted them. `note: none` checkpoints
      // and `_extra` rounds have no skeleton, so every number the student
      // typed was being listed as un-quoted against a note that does not
      // exist.
      const inFills = new Set(filledSlots.flatMap(slotTokens).filter(isFigure));
      const figuresDropped = markers.length
        ? [...new Set(answerish.flatMap(slotTokens).filter(isFigure))].filter(
            (f) => !inFills.has(f),
          )
        : [];
      const slotStrikes = slotDriftWarned.get(`${id}:slots`) ?? 0;
      // No note_markdown exemption: the renderer ignores note_markdown
      // whenever a skeleton exists, so taking that escape hatch discarded the
      // tutor's note and printed "*(not answered)*" under slots the student
      // had answered. Count NON-EMPTY fills — padding with "" satisfied a
      // length check and produced the same placeholder.
      // Only the slots the model fills. Demanding one for a «verbatim» slot
      // refused an honest call over a string the renderer throws away.
      const filled = modelSlotIdx.filter((i) => modelFill(i).trim()).length;
      // Not gated on markers.length: cp2_paperwork and cp4_shortcut_drawing
      // have exactly one marker and it is the model's, and it IS the keepsake
      // on those two — omitting it rendered "> **My cable:** done, sent it"
      // with no nudge at all.
      if (modelSlotIdx.length > 0 && filled < modelSlotIdx.length && slotStrikes < 2) {
        slotDriftWarned.set(`${id}:slots`, slotStrikes + 1);
        return toResult({
          out:
            `NOT LOGGED — this checkpoint's note needs ${modelSlotIdx.length} fill` +
            `${modelSlotIdx.length > 1 ? "s" : ""} from you and you sent ${filled}. Send note_slots ` +
            `in this order, one entry each:\n` +
            modelSlotIdx
              .map(
                (mi, n) =>
                  `  ${n + 1}. ${(markers[mi] ?? "").replace(/[«»]/g, "").replace(/\s+/g, " ").trim()}`,
              )
              .join("\n") +
            `\nThese are the parts no transcript holds — what the drawing shows, which ` +
            `option they picked, the numbers a widget displayed. Their typed words are ` +
            `quoted into the other slots for you. Then call checkpoint_done again.`,
          failed: false,
        });
      }
      // A skeleton that asks for the student's own words, on a checkpoint
      // where the student never typed any, means the question that was
      // supposed to get those words was skipped. cp1 is the case that
      // showed it: answered entirely through the picker, its reveal asks
      // one typed follow-up, and when the tutor jumped straight to
      // checkpoint_done the slot fell back to student_response and the
      // notebook read "**My guess:** about 60 — about 60". Photo and
      // drawing checkpoints are exempt by construction: their slots are
      // deliberately not marked «verbatim».
      if (said.length === 0 && markers.some((m) => /verbatim/i.test(m))) {
        problems.push(
          `this checkpoint's note quotes the student's own words, but they have not ` +
            `typed anything here yet — ask them the question your script's reveal_after ` +
            `names, in plain text, and wait for their answer`,
        );
      }
      // student_response is the headline quote of the graded record, so it
      // gets the same treatment as a «verbatim» slot — plus a repair the
      // slots do not need. A near-copy is snapped back to the transcript
      // silently (the correction is recorded in the log, not paraded at the
      // student), and only real invention — a figure or three content words
      // they never produced — is bounced back to the model.
      let responseSnappedFrom: string | null = null;
      // A bracketed stage direction is not a quote and must survive the snap.
      // Two checkpoints in this course pin one: `(no answer — moved on)`, and
      // cp0_welcome's `(nothing was asked — the session opened straight into
      // the story)`, which its script pins BECAUSE the record's headline is
      // the first thing a cold reader meets. The student still types
      // something at cp0 — "hi", "start" — so the snap replaced the pinned
      // literal with that word, and the graded record opened with
      // "start" as the answer to "(nothing asked — the session just opened)".
      // The script could not win against the repair; now it does not have to.
      const stageDirection = /^\((?:nothing|no answer)\b/i.test(response.trim());
      if (said.length > 0 && !stageDirection) {
        const snapped = snapToTranscript(response, said);
        if (snapped) {
          responseSnappedFrom = response;
          response = snapped;
        }
        const d = slotDrift(response, pool);
        if (d.numbers.length > 0 || d.words.length >= 3) {
          problems.push(
            `student_response ("${response.slice(0, 80)}") adds ` +
              [...d.numbers, ...d.words].map((t) => `"${t}"`).join(", "),
          );
        }
      }
      // When the answer came ONLY from a picker, student_response must be the
      // option they actually chose. This is the one case the transcript can't
      // police and the one a live run got backwards.
      if (said.length === 0 && picked.length > 0) {
        const pool = new Set(picked.flatMap(slotTokens));
        const shared = slotTokens(response).some((t) => pool.has(t) && !SLOT_GLUE.has(t));
        if (!shared) {
          problems.push(
            `student_response ("${response.slice(0, 80)}") does not match what they picked: ` +
              picked.map((p) => `"${p}"`).join(", "),
          );
        }
      }
      // Which message each «verbatim» slot appears to come from. RECORDED,
      // not enforced.
      //
      // The failure worth catching is a fill set shifted by one, which drops
      // the student's last answer from the keepsake. Two deterministic
      // guards were tried and both had to be withdrawn, because the thing
      // they must tell apart is not decidable without reading meaning:
      //   - "the last slot holds their last message" refused every
      //     checkpoint where the student's final message was a question or
      //     "ok its uploaded" — and told the tutor to move that aside INTO
      //     the graded slot;
      //   - "no two slots quote the same message" refused every checkpoint
      //     where the student answered two ask-steps in one breath and the
      //     tutor correctly split that message in two — including the
      //     module's own golden reference data.
      // A shift and a legitimate split look identical to any string
      // comparison. So the attribution goes in the log for a human to read,
      // the instruction lives in AGENTS.md, and nothing here refuses a
      // record on a guess.
      // Which of their messages the note quotes, and which it passed over.
      // (This used to be a per-slot best-bigram guess, from back when the
      // model wrote the quotes and could shift them by one. The quotes are
      // copied from the transcript now, so a guess would only mislead: what
      // a grader still wants is what was LEFT OUT.)
      const quotesMsgs = markers.some((m) => /verbatim/i.test(m)) ? quotedFrom : [];
      const skippedMsgs = markers.some((m) => /verbatim/i.test(m))
        ? said.map((_m, i) => i + 1).filter((n) => !quotedFrom.includes(n))
        : [];
      // A «verbatim» slot is filled from the transcript now, so the model's
      // fill for it is never rendered — policing it refused twice over a
      // string nobody reads and then stamped "⚠ Quoting check" on a note that
      // was character-perfect. What IS still the model's wording is a slot
      // that describes a picture, and that slot is only exempt while a
      // picture exists: when no photo arrived the answer was typed, and a
      // live run used the exemption to paraphrase a typed derivation.
      for (const i of photoMissing ? modelSlotIdx : []) {
        if (said.length === 0) break;
        const fill = modelFill(i);
        const d = slotDrift(fill, pool);
        if (d.numbers.length === 0 && d.words.length < 3) continue;
        problems.push(
          `note slot ${i + 1} ("${fill.slice(0, 80)}") adds ` +
            [...d.numbers, ...d.words].map((t) => `"${t}"`).join(", "),
        );
      }

      // A hand-written note is the one place words can be put in the
      // student's mouth. Every other note cell is a skeleton whose «verbatim»
      // slot the extension fills from the transcript — but a checkpoint with
      // no skeleton (a practice round, a script with no `note:`) takes
      // note_markdown, written by the model, blockquote and all. A live run
      // wrote `> "P is 3, Q is 3, R is 2"` under "I worked out", while the log
      // beside it held the student's three actual sentences. Nobody had said
      // the quoted line. Same test as the slots, on the blockquote lines only:
      // the prose above the fold is the tutor's own and is not checked.
      if (!noteSuppressed(id) && !noteSkeleton(id) && said.length > 0) {
        const quoted = String(params.note_markdown ?? "")
          .split("\n")
          .filter((l) => /^\s*>/.test(l))
          .map((l) => l.replace(/^\s*>\s?/, ""))
          .join(" ")
          .trim();
        if (quoted) {
          const d = slotDrift(quoted, pool);
          if (d.numbers.length > 0 || d.words.length >= 3) {
            problems.push(
              `the quote in note_markdown ("${quoted.slice(0, 80)}") adds ` +
                [...d.numbers, ...d.words].map((t) => `"${t}"`).join(", ") +
                ` — quote their turns as they typed them, joined with " · "`,
            );
          }
        }
      }

      // Two refusals per checkpoint, then it logs anyway with the drift
      // flagged for the grader: a model that cannot satisfy the check must
      // never be able to strand the student mid-lesson.
      const strikes = slotDriftWarned.get(id) ?? 0;
      if (problems.length > 0 && strikes < 2 && said.length === 0) {
        slotDriftWarned.set(id, strikes + 1);
        return toResult({
          out:
            `NOT LOGGED — ${problems.join("; ")}.\n` +
            `Do not write that line for them and do not call checkpoint_done again ` +
            `until they have answered. SAY THE QUESTION OUT LOUD in plain text — never ` +
            `end this turn without speaking, a silent turn is a frozen screen — then ` +
            `wait, and log what they type.`,
          failed: false,
        });
      }
      if (problems.length > 0 && strikes < 2) {
        slotDriftWarned.set(id, strikes + 1);
        return toResult({
          out:
            `NOT LOGGED — student_response and every note slot quoting them are the ` +
            `student's own words (a photo slot too, when no photo arrived and the answer ` +
            `was typed), and these are not in anything they said: ` +
            `${problems.join("; ")}.\n` +
            `What they actually typed: ${said.map((s) => `"${s}"`).join(", ")}.\n` +
            `Quote them, don't polish — no figure they didn't give, no sentence built out ` +
            `of your own summary. Your reading of a drawing, or a number a widget showed, ` +
            `belongs in notes.` +
            (picked.length
              ? ` They picked: ${picked.map((p) => `"${p}"`).join(", ")} — record THAT.`
              : "") +
            ` Then call checkpoint_done again.`,
          failed: false,
        });
      }
      // ── The close that came too late ────────────────────────────────────
      // The verbatim capture resets HERE and nowhere else, so a close that
      // waits until the next question is already being answered files two
      // checkpoints' words under one id. A gate run did exactly that: the
      // student was walked pair-by-pair through cp2_distance, then answered
      // "can you just tell me the answer?" and cp2_diameter's "the A-D one is
      // the tallest bar" — and all nine lines, both checkpoints', landed in
      // cp2_distance's row, logged `pass` with zero hints. cp2_diameter's own
      // row kept a single line and lost the hint it was given. Nothing in the
      // record said any of this happened.
      //
      // The turn count is the tell. A checkpoint answered when it is asked
      // closes in a handful of turns; the ⚖️ nudge already treats STUCK_TURNS
      // as "these two are going round". Zero hints after that many turns is
      // not a clean pass, whatever the model believes it remembers. Refuse
      // once — the model gets to recount, and to close the next one on time —
      // then log whatever comes back, because a guard that will not let go is
      // a student stuck at a checkpoint they have already finished.
      // The same zero, caught earlier. The turn count above only fires after
      // twelve turns, and a checkpoint can be guided through in five: two live
      // runs logged `pass` with no hints on rows whose own verbatim array
      // holds the wrong first answer that a hint corrected, and one of them
      // wrote "invented it unprompted" in the notes beside it.
      //
      // The tell that IS in reach: a scripted checkpoint asks a known number
      // of questions, and every extra message the student sent is one the
      // tutor asked for. More answers than questions, with no hints, is worth
      // one look. Conservative on purpose — the count is taken only from an
      // `ask` block that numbers its questions, a multi-part answer typed as
      // two messages costs one extra tool call and nothing else, and the
      // nudge says so.
      const qCount = scriptedQuestionCount(baseCheckpointId(id));
      const moreAnswersThanQuestions = qCount > 0 && said.length > qCount;
      const lateStrikes = lateCloseWarned.get(id) ?? 0;
      if (
        (turnsInCheckpoint >= STUCK_TURNS || moreAnswersThanQuestions) &&
        Math.round(Number(params.hints_used ?? 0) || 0) === 0 &&
        judgment !== "prediction" &&
        lateStrikes < 1
      ) {
        lateCloseWarned.set(id, lateStrikes + 1);
        return toResult({
          out:
            `NOT LOGGED — ` +
            (moreAnswersThanQuestions
              ? `this checkpoint's script asks ${qCount} question${qCount === 1 ? "" : "s"} ` +
                `and the student sent ${said.length} answers, `
              : `this checkpoint has been open for ${turnsInCheckpoint} turns `) +
            `and you are logging it with no hints. Two things to fix, in this order.\n` +
            `1. hints_used: count the smaller questions you asked to get them there. ` +
            `A question you asked because an answer was not there yet is a hint, even ` +
            `if they got it on the very next turn — and "unprompted" does not belong ` +
            `in the notes of a row like that. Hints are never held against the ` +
            `student; a wrong count is what damages the record. If they really did ` +
            `answer everything first time (two messages for one two-part answer, say), ` +
            `send it again unchanged and it will log.\n` +
            `2. If you have already moved on to the next question, this close is late, ` +
            `and everything they typed since — including their answer to the NEXT ` +
            `checkpoint — is about to be filed under this one. Close a checkpoint the ` +
            `moment its answer lands, before you ask the next thing.\n` +
            `Then call checkpoint_done again. It will log whatever you send this time.`,
          failed: false,
        });
      }
      // The next scripted checkpoint is now the open one, even if it builds
      // nothing — that is what keeps its note cell ahead of the next build.
      // Off-script ids (a stretch, a typo) must NOT null it out: that would
      // switch the ordering guard off for the rest of the session.
      // Monotonic. Closing an out-of-order practice round — a
      // `cp2_distance_extra` logged from chapter 3 — used to move the open
      // checkpoint BACKWARDS, and the build guard it arms then refused every
      // remaining build with "close cp2_paperwork first": a checkpoint that
      // already has a row and a note, so obeying wrote a duplicate into the
      // graded record. The session's position only ever goes forward.
      if (isScriptedCheckpoint(id)) {
        const ord = checkpointOrder();
        const advanced = nextCheckpointId(id);
        // null means "module finished" — the END of the order, not before
        // its start, or closing a stray practice round could pull the
        // session back out of the terminal state.
        const here = pendingCheckpoint ? ord.indexOf(pendingCheckpoint) : ord.length;
        const there = advanced ? ord.indexOf(advanced) : ord.length;
        if (there >= here) pendingCheckpoint = advanced;
      }
      // The resume window closes with the first checkpoint: past this point
      // every dialog answer is a lesson answer.
      awaitingResumeChoice = false;
      studentSaidSince(ctx, true);
      pickedSince(true);
      // The detour marks belong to the checkpoint just closed — the same
      // words typed again later are a fresh answer.
      detourAsked.clear();

      // A checkpoint that needed hints is `pass_with_hints`, whatever the
      // model typed. It logged plain `pass` with hints_used 2 in a live run —
      // not dishonesty, just two fields the model has to keep agreeing, so
      // the extension makes them agree. Hints are never penalised; the point
      // is that the record says what happened.
      let hintsOut = Math.max(0, Math.round(Number(params.hints_used ?? 0) || 0));
      // The two fields must agree, whichever way round they were typed. A
      // `guided` checkpoint is by definition one the tutor walked them
      // through, so zero hints there is a miscount, not a cleaner record.
      const judgmentOut =
        judgment === "pass" && hintsOut > 0
          ? "pass_with_hints"
          : judgment === "pass_with_hints" && hintsOut === 0
            ? "pass"
            : judgment;
      if (judgmentOut === "guided" && hintsOut === 0) hintsOut = 1;
      // A row logged under a referee ruling says so, for the graders. The
      // waiver is spent here — except next_chapter, which chapter_done's own
      // gate must still see and consume.
      let appealRuling: string | null = null;
      if (refereeWaiverActive()) {
        appealRuling = refereeWaiver!.ruling;
        if (appealRuling !== "next_chapter") refereeWaiver = null;
      }
      turnsInCheckpoint = 0; // a closed checkpoint is not a stuck one
      const logged = appendLog({
        type: "checkpoint",
        id,
        question: String(params.question ?? ""),
        student_response: response,
        judgment: judgmentOut,
        hints_used: hintsOut,
        notes: String(params.notes ?? ""),
        student_said_verbatim: said,
        ...(picked.length > 0 ? { student_picked: picked } : {}),
        ...(responseSnappedFrom ? { response_retyped_as: responseSnappedFrom } : {}),
        ...(photoMissing ? { photo_missing: true } : {}),
        // The build guard gives up after two refusals and logs anyway — the
        // right call, since a guard that can strand a student is worse than
        // the fault it catches. But it left no mark, so a row for a checkpoint
        // whose figure is not in the notebook was byte-identical to an honest
        // one, while the photo gate beside it has always stamped its own gap.
        ...(missingBuild.length ? { build_missing: missingBuild } : {}),
        // Which id the model actually sent, when it was not the script's.
        ...(snapped.snappedFrom ? { id_snapped_from: snapped.snappedFrom } : {}),
        ...(appealRuling ? { closed_by_referee: appealRuling } : {}),
        ...(quotesSnapped.length ? { slot_quotes_repaired: quotesSnapped } : {}),
        ...(quotesMsgs.length ? { note_quotes_msgs: quotesMsgs } : {}),
        ...(skippedMsgs.length ? { note_skipped_msgs: skippedMsgs } : {}),
        ...(figuresDropped.length ? { figures_not_quoted: figuresDropped } : {}),
        ...(problems.length > 0 ? { verbatim_drift: problems } : {}),
      });

      const suppressed = noteSuppressed(id);
      const skeleton = suppressed ? "" : noteSkeleton(id);
      const md = suppressed
        ? ""
        : skeleton
          ? fillSlots(skeleton, filledSlots, markers.length === 1 ? response : "")
          : String(params.note_markdown ?? "").trim();
      let noteLine: string;
      if (suppressed) {
        noteLine =
          `No note cell for this one, by design (the script says note: none) — it is ` +
          `logged and appears in the closing record. Do NOT add one yourself.`;
      } else if (!md) {
        noteLine =
          `NO NOTE CELL: this checkpoint has no note: skeleton and you passed no ` +
          `note_markdown — add one now with nb_add_cell (name "${id}_note").`;
      } else {
        // Park FIRST, unpark on success. The row is already on disk by this
        // point, and inserting a cell takes a round trip to the kernel — so a
        // session that ends inside that round trip (Ctrl-C, a closed terminal,
        // a shut laptop, all of them ordinary ways for a beginner to stop)
        // left a logged checkpoint whose note existed nowhere: not in the
        // notebook, not parked, and not in the model's context after the next
        // compaction. Written down before the risky part, it survives.
        await flushParkedNotes(signal);
        parkNote(`${id}_note`, md);
        const r = await insertMarkdownCell(`${id}_note`, md, signal);
        // A note that could not be written stays PARKED, exactly as rendered.
        // In a live run the notebook died mid-checkpoint, and by the time the
        // tutor came back to write the note the student's wording had been
        // compacted out of its context — so it reconstructed the quotes from
        // memory and paraphrased three of them into the graded artifact. The
        // text is already correct here; nothing needs to remember it.
        if (!r.failed) unparkNote(`${id}_note`);
        noteLine = r.failed
          ? `Note cell PARKED — the notebook is unreachable, so the note is saved ` +
            `exactly as written and goes in by itself the moment the notebook is back. ` +
            `Do NOT rewrite it later from memory; keep teaching.`
          : `Note cell added.`;
      }

      // Every branch NAMES the checkpoint to go to. Without that, a small
      // model three practice rounds deep loses the thread: in a live run it
      // answered READY by inventing yet another warm-up and logging
      // cp0_welcome_extra, then _extra_extra — the student could not leave
      // cp0 at all. "Start the next checkpoint from your script" is not an
      // instruction a flash model can act on; "start cp1_milgram" is.
      const nextId = nextCheckpointId(id);
      // "Find it in your CHAPTER SCRIPT" is only true while the next id is in
      // THIS chapter. At every chapter boundary it named a checkpoint the
      // script in context does not contain, contradicting the script's own
      // closing line ("call chapter_done") in the same turn — and the tool
      // result is the instruction a flash model actually follows.
      const thisChapter = loadChapters().find((c) => c.checkpoints.includes(baseCheckpointId(id)));
      const nextIsHere = !!nextId && !!thisChapter && thisChapter.checkpoints.includes(nextId);
      const goNext = !nextId
        ? `That was the last checkpoint of the module — call chapter_done next.`
        : nextIsHere
          ? `Start checkpoint "${nextId}" NOW: find it in your CHAPTER SCRIPT and ask its ` +
            `first question. Do not revisit "${baseCheckpointId(id)}".`
          : `That was the last checkpoint of this chapter — call chapter_done next. It ` +
            `loads the chapter that holds "${nextId}"; do not start that checkpoint from ` +
            `memory.`;
      const READY = "Ready for the next question";
      const ASK_Q = "I have a question first";
      const MORE = "Give me another one like that";
      // When the picker cannot run — a dead notebook, a headless restart —
      // the pace gate becomes a sentence, and a live run simply carried on
      // into the next checkpoint without ever asking. So the sentence leads
      // with the prohibition, and the next build is bounced once to make it
      // stick.
      let nextLine =
        `No picker available. Do NOT start the next checkpoint yet: ask the student in ` +
        `plain text whether they are ready, END YOUR TURN, and wait for their answer. ` +
        `Only when they say yes: ${goNext}`;
      let paceAsked = false;
      if (ctx?.ui?.select) {
        paceAsked = true;
        // The reveal, above the question, so the beat survives a short window.
        const said = lastTutorLine(ctx);
        const { choice, typed } = await askStudent(
          ctx,
          said ? `${said}\n\nWhere to next?` : "Where to next?",
          [READY, ASK_Q, MORE],
        );
        const practiceRound = /_extra/.test(id);
        nextLine =
          choice === READY
            ? `The student is READY. ${goNext}`
            : choice === ASK_Q
              ? `The student has a QUESTION. Do NOT advance: ask what it is, answer it ` +
                `properly, leave a souvenir cell, call log_detour, then ask them again ` +
                `in plain text whether to move on. When they are ready: ${goNext}`
              : choice === MORE
                ? // A checkpoint with no fresh_variants has nothing to practise
                  // (cp0 is a calibration question), and a second helping of an
                  // already-repeated one is where the loop started.
                  !hasFreshVariants(baseCheckpointId(id))
                  ? `The student asked for MORE PRACTICE, but this checkpoint has no ` +
                    `practice variant — it is not that kind of question. Say so warmly in ` +
                    `ONE sentence, then: ${goNext}`
                  : practiceRound
                    ? `They have already had a practice round here. Give at most one more, ` +
                      `then move on regardless: ${goNext}`
                    : `The student wants MORE PRACTICE. Do NOT advance yet: improvise ONE ` +
                      `problem of the same kind on NEW data, from this checkpoint's ` +
                      `fresh_variants. Guide, then checkpoint_done again with id ` +
                      `"${baseCheckpointId(id)}_extra" (never a fail). After that: ${goNext}`
                : choice === TYPE_IT && typed
                  ? `Do NOT start the next checkpoint. The student typed this instead of ` +
                    `picking a row — these are THEIR words, react to them: "${typed}". ` +
                    `Answer whatever it asks (a question gets a proper answer, a souvenir ` +
                    `cell and log_detour), then ask in plain text whether they are ready. ` +
                    `Only when they say yes: ${goNext}`
                  : `The student closed the picker — ask in plain text what they'd like to do. ` +
                    `If they want to move on: ${goNext}`;
      }

      if (!paceAsked && nextId) paceUnasked.add(nextId);
      return toResult({
        out:
          `Logged${logged ? "" : " (LOG WRITE FAILED — tell no one, keep teaching)"}. ` +
          `${noteLine}\n${nextLine}`,
        failed: false,
      });
    },
    ...quiet("Writing that into your notebook…"),
  });

  // ── nb_submit ─────────────────────────────────────────────────────────────
  //
  // Handing the work in, from inside the session.
  //
  // This did not exist, and its absence cost a student their submission. They
  // asked the tutor mid-session to put their work on GitHub; the tutor has no
  // shell and no instruction covering it, so it refused. They appealed to the
  // referee, which has six rulings and none of them is "submit" — so it ruled
  // on the nearest thing, the tutor accepted the ruling warmly, and the
  // session moved on to the next chapter with the request simply gone. The
  // work sat unsubmitted in a folder.
  //
  // There is a second half nobody would have found by reading the module:
  // these assignments are `submission_mode: "tag"`, where a plain `git push`
  // is DELIBERATELY not a submission — classroom50 suppresses branch-triggered
  // runs and grades only a pushed `submit/*` tag. The README told students to
  // push the branch. So even a student who did it by hand, exactly as
  // documented, was not handing anything in. Both halves are why this tool
  // tags as well as pushes.
  //
  // git is run directly, never through a shell: the tutor's bash tool is
  // excluded on purpose, and on Windows the student has Git Bash but the
  // agent does not.
  const gitRun = (
    args: string[],
    opts: { input?: string } = {},
  ): { ok: boolean; out: string } => {
    try {
      const r = spawnSync("git", args, {
        cwd: process.cwd(),
        encoding: "utf-8",
        input: opts.input,
        timeout: 120_000,
        // No credential prompt may ever block the session waiting on a
        // terminal the student is not looking at.
        env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
      });
      const out = `${r.stdout ?? ""}${r.stderr ?? ""}`.trim();
      return { ok: r.status === 0, out };
    } catch (e: any) {
      return { ok: false, out: String(e?.message ?? e) };
    }
  };

  /** `submit/<UTC>-<short sha>` — classroom50's canonical namespace, minted
   *  the same way its own workflow mints it, so a pushed tag IS the record. */
  const submissionTag = (sha: string): string => {
    const t = new Date().toISOString().replace(/\.\d+Z$/, "Z").replace(/:/g, "-");
    return `submit/${t}-${sha.slice(0, 7)}`;
  };

  pi.registerTool({
    name: "nb_submit",
    label: "Hand in",
    description:
      "Commit the student's work and hand it in. Call this WHENEVER they ask to submit, " +
      "hand in, save to GitHub, or push — at any point in the session, not only at the " +
      "end — and once more when the module finishes. Never refuse the request and never " +
      "tell them to type git commands: this is the tool for it. It commits everything " +
      "(the notebook, the session log, their photos and code), pushes, and then pushes a " +
      "submission tag, which is what actually starts the grading. Safe to call twice; " +
      "with nothing new to commit it just says so.",
    promptSnippet: "Commit and hand in the student's work",
    parameters: Type.Object({
      status: STATUS_PARAM,
      message: Type.Optional(
        Type.String({
          description:
            "Commit message. Something plain about where they are — " +
            "\"Pair Notebook 01 — through chapter 3\". Defaults to the module and the date.",
        }),
      ),
    }),
    execute: async (_id: any, params: any, _signal: any, _onUpdate: any) => {
      const inside = gitRun(["rev-parse", "--is-inside-work-tree"]);
      if (!inside.ok) {
        return toResult({
          out:
            `NOT HANDED IN — this folder is not a git repository, so there is nothing to ` +
            `push to. Their work is still safe on disk (the notebook and ` +
            `session_artifacts/ are written as you go). Tell them plainly that the ` +
            `folder they are working in is not the one the assignment link created, and ` +
            `that their instructor can sort it out — then carry on teaching.`,
          failed: false,
        });
      }

      const msg =
        String(params?.message ?? "").trim() ||
        `Pair Notebook — ${new Date().toISOString().slice(0, 10)}`;

      gitRun(["add", "-A"]);
      const staged = gitRun(["diff", "--cached", "--quiet"]);
      let committed = false;
      if (!staged.ok) {
        let c = gitRun(["commit", "-m", msg]);
        if (!c.ok && /please tell me who you are|user\.email|user\.name/i.test(c.out)) {
          // A fresh machine has no git identity, and "Please tell me who you
          // are" is not something to hand a student mid-lesson. Commit as the
          // repo's own owner, the way GitHub's web editor does.
          const url = gitRun(["remote", "get-url", "origin"]).out;
          const owner = /[/:]([^/]+)\/[^/]+?(?:\.git)?$/.exec(url)?.[1] ?? "student";
          c = gitRun([
            "-c", `user.name=${owner}`,
            "-c", `user.email=${owner}@users.noreply.github.com`,
            "commit", "-m", msg,
          ]);
        }
        if (!c.ok) {
          return toResult({
            out:
              `NOT HANDED IN — the commit failed: ${c.out.slice(0, 300)}\n` +
              `Say in one warm line that you could not hand it in and their instructor ` +
              `will sort it out. Their work is still on disk. Then carry on teaching.`,
            failed: false,
          });
        }
        committed = true;
      }

      const sha = gitRun(["rev-parse", "HEAD"]).out.trim();
      const branch = gitRun(["rev-parse", "--abbrev-ref", "HEAD"]).out.trim() || "HEAD";
      const hasRemote = gitRun(["remote", "get-url", "origin"]).ok;
      if (!hasRemote) {
        return toResult({
          out:
            `Committed${committed ? "" : " (nothing new to commit)"}, but NOT handed in: ` +
            `this folder has no GitHub remote, so there is nowhere to push. Tell them ` +
            `their work is saved here and their instructor can sort the hand-in out.`,
          failed: false,
        });
      }

      const push = gitRun(["push", "origin", `HEAD:${branch}`]);
      if (!push.ok) {
        return toResult({
          out:
            `Committed, but the push failed: ${push.out.slice(0, 300)}\n` +
            `Their work IS saved and committed — nothing is lost. Say in one line that ` +
            `you could not reach GitHub and they should tell their instructor, then carry ` +
            `on teaching. Do not debug it with them and do not ask them to type anything.`,
          failed: false,
        });
      }

      // The tag is the submission. Without it a push in tag mode is filed and
      // never graded, which is the failure this whole tool exists for.
      //
      // But only one tag per commit. A student who asks twice with nothing
      // changed in between would otherwise get a second tag on the same sha,
      // and each one starts its own grading run over identical work —
      // classroom50 dedupes by sha on its side, so it is not wrong, just
      // noise nobody asked for. Already tagged means already handed in, and
      // saying so is a better answer than doing it again.
      const existing = gitRun(["tag", "--points-at", "HEAD", "--list", "submit/*"])
        .out.split("\n").map((s) => s.trim()).filter(Boolean);
      let tag = existing[existing.length - 1] ?? "";
      let madeTag = !!tag;
      if (!tag) {
        tag = submissionTag(sha);
        madeTag = gitRun(["tag", tag]).ok && gitRun(["push", "origin", tag]).ok;
      }

      return toResult({
        out:
          (committed
            ? `Handed in. `
            : existing.length
              ? `Already handed in — nothing has changed since last time. `
              : `Handed in (nothing new to commit — pushed what was there). `) +
          (madeTag
            ? `Submission tag ${tag} is what starts the grading.`
            : `WARNING: the push worked but the submission tag did not, so this may not ` +
              `be graded — tell them to mention it to their instructor.`) +
          `\nSay it in ONE short line — "that's handed in" — and go straight back to the ` +
          `lesson. They can carry on and hand in again as often as they like.`,
        failed: false,
      });
    },
    ...quiet("Handing your work in…"),
  });

  // ── log_detour ────────────────────────────────────────────────────────────
  pi.registerTool({
    name: "log_detour",
    label: "Log detour",
    description:
      "Record a student question you answered off-script (their curiosity is graded as " +
      "engagement) and leave the souvenir in their notebook. Build the souvenir with " +
      "nb_add_cell first — text AND a picture in one mo.vstack, or a playable demo — and " +
      "pass its cell_name. souvenir_markdown is the fallback for an idea no picture helps.",
    promptSnippet: "Log a student's off-script question and leave a souvenir cell",
    parameters: Type.Object({
      status: STATUS_PARAM,
      question: Type.String({
        description:
          "Their question VERBATIM — but the QUESTION only. If they answered and " +
          "asked in one breath (\"2 out of 10 - but why does that matter?\"), send " +
          "just the question part; the rest is their answer and belongs in the note.",
      }),
      what_you_did: Type.String({ description: "One line: how you answered it." }),
      souvenir_markdown: Type.Optional(
        Type.String({
          description:
            "Fallback only: markdown for a text-only 🧭 Detour cell. Prefer nb_add_cell " +
            "+ cell_name, so the souvenir has something to look at or play with.",
        }),
      ),
      cell_name: Type.Optional(
        Type.String({ description: "Name of the souvenir cell you already added." }),
      ),
    }),
    async execute(_id, params, signal, _onUpdate, ctx: any) {
      const question = String(params.question ?? "").trim();
      const md0 = String(params.souvenir_markdown ?? "").trim();
      const cellName = String(params.cell_name ?? "").trim();
      // A souvenir is a keepsake, not a log line: the notebook's whole promise
      // is that a curious student's copy looks different from everyone else's,
      // and four text-only blobs in one live session is what that failure
      // looks like. Bounced ONCE, with both ways out spelled out, then
      // accepted — some ideas really are just words.
      //
      // The cell the tutor built ITSELF is checked the same way. Trusting
      // that path was the hole: every detour in the failing session arrived
      // as a cell_name, and every one of them was prose that never quoted
      // the question it answered.
      // The question as the STUDENT typed it, not as the tutor remembers it.
      // A live run logged a tidied paraphrase — "wait whats this called," gone,
      // "it" reworded — and the souvenir quoted that.
      const saidNow = studentSaidSince(ctx, false);
      const snapped = snapToTranscript(question, saidNow);
      let snappedQ = snapped ?? question;
      // Mark the message they asked it in, so the next checkpoint's note
      // quotes their answer and not their question — the souvenir already
      // holds the question, word for word.
      //
      // NOT gated on `snapped`: snapToTranscript is a REPAIR, and its first
      // act is to return null when the text is already in the transcript
      // because there is nothing to repair. Gating on it meant the dedup
      // fired only when the model DISOBEYED "their question VERBATIM" and
      // never when it obeyed — measured: verbatim and true paraphrase both
      // leaked, a tidied quote was the only shape that worked.
      {
        const qNorm = normMsg(question);
        // A message containing the question is a match; a message CONTAINED
        // IN it is not — "triangles" sits inside "why do triangles matter?"
        // and is a perfectly good answer.
        let bestIdx = -1;
        let bestScore = 0;
        saidNow.forEach((m, i) => {
          const n = normMsg(m);
          const d = n === qNorm || (qNorm.length > 0 && n.includes(qNorm)) ? 1 : bigramDice(m, question);
          if (d > bestScore) {
            bestScore = d;
            bestIdx = i;
          }
        });
        // Below near-identity, the message also has to LOOK like a question
        // — and question shape is decided at the START of the utterance, not
        // by a wh-word anywhere in it. A wh-word ANYWHERE deleted a student's
        // real answer from their own note: "clustering is how often my
        // friends know each other" scores 0.78 against the question "how
        // often do friends know each other", and contains "how". Dropping
        // their work from the graded artifact is far worse than leaving a
        // question in it, so this errs toward keeping.
        // `which`, `whether` and `if` are deliberately NOT openers: they lead
        // declaratives at least as often ("which means C stays the same").
        const asked = bestIdx >= 0 ? saidNow[bestIdx] : "";
        const QUESTION_OPENER =
          /^(?:(?:wait|hang on|hold on|sorry|ok|okay|hmm+|umm+|oh|but|and|also|quick|one more|side note|random|off topic)[ ]+)*(?:what|whats|why|how|hows|when|where|who|whos|is|are|isnt|arent|do|does|did|dont|doesnt|can|could|should|would|will)\b|^(?:i (?:dont|do not|didnt|cant) (?:get|understand|see|follow)|im (?:confused|lost)|i(?:m| am) not sure (?:what|why|how))\b/;
        const looksAsked = /\?/.test(asked) || QUESTION_OPENER.test(normMsg(asked));
        // A message that is their question AND something else keeps its
        // something else. Dropping whole messages cost a student the "i
        // count 2" out of "how many could there be? i count 2" — half an
        // answer, gone from the note, on the checkpoint that was about
        // counting. Any figure in the residue, or three content words of
        // it, means this message is carrying more than the question.
        const qNormFull = normMsg(question);
        const aNorm = normMsg(asked);
        // WHERE the leftovers sit decides what they are. A lead-in runs
        // BEFORE the question ("wait, quick question before I do the average
        // — does it matter which direction…"), and a live run lost that whole
        // checkpoint's note to it: four content words in the run-up read as
        // an answer. What comes AFTER is the half that is usually an answer.
        // Either way a figure anywhere in the leftovers means graded content,
        // and the message stays whole.
        const at = qNormFull ? aNorm.indexOf(qNormFull) : -1;
        const tokensOf = (t: string) => t.split(" ").filter(Boolean);
        const before = at >= 0 ? tokensOf(aNorm.slice(0, at)) : [];
        const after = at >= 0 ? tokensOf(aNorm.slice(at + qNormFull.length)) : [];
        const qTok = new Set(tokensOf(qNormFull));
        const residue =
          at >= 0 ? [...before, ...after] : tokensOf(aNorm).filter((t) => !qTok.has(t));
        // Only the trailing half is weighed by length; a lead-in is judged on
        // figures alone.
        const weighed = at >= 0 ? after : residue;
        const HEDGE_WORDS = new Set([
          "wait", "hang", "hold", "sorry", "ok", "okay", "hmm", "hmmm", "umm", "um",
          "oh", "ohh", "quick", "random", "side", "topic", "off", "exactly", "just",
          "really", "actually", "again", "please", "btw", "like", "though", "tho",
          "whats", "hows", "whos", "one", "more", "note", "er",
        ]);
        const carriesMore =
          residue.some((t) => /\d/.test(t)) ||
          weighed.filter((t) => !HEDGE_WORDS.has(t) && !SLOT_GLUE.has(t)).length >= 3;
        if (
          bestIdx >= 0 &&
          !carriesMore &&
          (bestScore >= 0.9 || (bestScore >= 0.6 && looksAsked))
        ) {
          detourAsked.add(normMsg(asked));
          // Whatever the note leaves out, the souvenir says in full — and in
          // THEIR words. A live run logged the question with the student's
          // lead-in trimmed off ("Wait, quick question first —"), and the
          // souvenir quoted the trim. snapToTranscript cannot repair that: a
          // question the message CONTAINS is, to a repair function, already
          // fine. We have just decided this message is the question, so use
          // it.
          if (normMsg(asked) !== normMsg(snappedQ)) snappedQ = asked;
        }
        if (snapped) detourAsked.add(normMsg(snapped));
      }
      let gap = "";
      if (cellName) {
        const src = await readCellSource(cellName, signal);
        // The quote line is the extension's job, not the model's. Left to the
        // model it tidied the student's punctuation ("whats" → "what's", a
        // "?" appended) — a small edit to the one thing in the cell that is
        // supposed to be theirs, word for word. It goes in unless their exact
        // words are already there, and the gap check no longer asks the model
        // for it at all.
        if (src !== null && src.trim() === "") {
          // No such cell. Saying "is prose only" about a cell that does not
          // exist sent the model editing thin air; the honest gap is the one
          // withQuotedQuestion's sibling check has always used.
          gap = "does not exist in the notebook";
        } else if (src !== null) {
          // Words, not bytes — the SAME normalisation withQuotedQuestion uses.
          // A byte-exact test called a cell unquoted because the quote had
          // typographic apostrophes, and the cell then carried "You asked"
          // twice: once raw, once tidied.
          const flat = (x: string) =>
            x
              .toLowerCase()
              .replace(/['\u2018\u2019\u02BC"\u201C\u201D]/g, "")
              .replace(/[^a-z0-9]+/g, " ")
              .trim();
          // Either wording: the model wrote this cell against its own
          // `question`, and snappedQ may since have grown into the student's
          // full message.
          let quoted = flat(src).includes(flat(snappedQ)) || flat(src).includes(flat(question));
          if (!quoted) quoted = await prependQuestionToCell(cellName, snappedQ, signal);
          const shows =
            /netviz\s*\(|mo\.ui\.|mo\.image\s*\(|alt\.Chart|sns\.\w+\s*\(|plt\.\w+\s*\(/.test(src);
          gap = !shows
            ? "is prose only — nothing to look at or play with"
            : // The prepend failed (a cell marimo will not rewrite). Reporting
              // success here shipped a souvenir with no question in it.
              quoted
              ? ""
              : "never quotes the question it answers";
        }
      }
      const gapKey = cellName || question;
      if (gap && (detourTextOnlyWarned.get(gapKey) ?? 0) < 2) {
        detourTextOnlyWarned.set(gapKey, (detourTextOnlyWarned.get(gapKey) ?? 0) + 1);
        // "Fix it with nb_edit_cell" is the right advice for a cell that is
        // there and thin. For a cell that does not exist it is impossible
        // advice: nb_edit_cell cannot create one, so it prints "no cell named
        // X", which renders as a ✓, and the model tries again against the
        // same nothing until the strikes run out.
        const missing = gap === "does not exist in the notebook";
        return toResult({
          out: missing
            ? `NOT LOGGED YET — there is no cell named "${cellName}" in the notebook. ` +
              `nb_edit_cell cannot make one; build it first with nb_add_cell (name ` +
              `"${cellName}"), text and picture in ONE cell —\n` +
              `  mo.vstack([mo.md(r"""> 🧭 **You asked:** “…”\n\n…"""), netviz(edges, highlight=[…])])\n` +
              `— or an nb_add_exercise box if the idea is playable. Then call log_detour ` +
              `again with the same cell_name.`
            : `NOT LOGGED YET — the souvenir cell "${cellName}" ${gap}. Fix it with ` +
              `nb_edit_cell so it holds something to see or try (their question is ` +
              `quoted for you, you do not need to add it) —\n` +
              `  mo.vstack([mo.md(r"""> 🧭 **You asked:** “…”\n\n…"""), netviz(edges, highlight=[…])])\n` +
              `— or an nb_add_exercise box if the idea is playable. Then call log_detour ` +
              `again with the same cell_name.\n` +
              `If words genuinely are the whole answer here, call log_detour again as it ` +
              `is and it will be accepted.`,
          failed: false,
        });
      }
      if (!cellName && md0 && (detourTextOnlyWarned.get(question) ?? 0) < 2) {
        detourTextOnlyWarned.set(question, (detourTextOnlyWarned.get(question) ?? 0) + 1);
        return toResult({
          out:
            `NOT LOGGED YET — a text-only souvenir is the weakest kind of keepsake. Build ` +
            `the cell first with nb_add_cell (name "detour_<topic>"), text and picture in ` +
            `ONE cell: mo.vstack([mo.md(r"""…"""), netviz(edges, highlight=[…])]) — or an ` +
            `nb_add_exercise box if the idea is something they can try. Then call ` +
            `log_detour again with cell_name.\n` +
            `If a picture genuinely adds nothing to this one, call log_detour again with ` +
            `the same souvenir_markdown and it will be accepted as it is.`,
          failed: false,
        });
      }
      // Insert BEFORE logging, so the row names the cell it actually made. It
      // logged cell:"" on this path, and a grader auditing souvenirs from the
      // log found none for a question that has one.
      let madeCell = "";
      let madeFailed = false;
      if (!cellName && md0) {
        const md = withQuotedQuestion(md0, snappedQ, question);
        const slug = sanitize(question.toLowerCase().split(/\s+/).slice(0, 4).join("_")).slice(
          0,
          40,
        );
        const name = `detour_${slug || "note"}`;
        const r = await insertMarkdownCell(name, md, signal);
        madeFailed = r.failed;
        if (!r.failed) madeCell = name;
      }
      appendLog({
        type: "detour",
        question,
        what_you_did: String(params.what_you_did ?? ""),
        cell: String(params.cell_name ?? "") || madeCell,
        // Peek, never consume: a detour normally happens mid-checkpoint, and
        // consuming the mark here would move the student's checkpoint answer
        // into the detour record — leaving checkpoint_done with an empty
        // student_said_verbatim and a drift check with nothing to match.
        student_said_verbatim: studentSaidSince(ctx, false),
        // Accepted on the retry despite the gap — the grader sees what the
        // souvenir was missing rather than the tool pretending it was fine.
        ...(gap ? { souvenir_gap: gap } : {}),
      });
      if (!md0) {
        return toResult({
          out: cellName
            ? // Do not call a cell that is not there "noted": the give-up path
              // reported success for a souvenir the student's notebook does
              // not contain, and the model moved on satisfied.
              gap === "does not exist in the notebook"
              ? `Logged — but there is still NO cell named "${cellName}" in the notebook, ` +
                `so this detour has no souvenir. Build it with nb_add_cell (not ` +
                `nb_edit_cell, which cannot create a cell) when you get a moment.`
              : `Logged. Souvenir cell "${cellName}" noted.`
            : `Logged — but NO souvenir cell yet. Add one now (nb_add_cell, name ` +
              `"detour_<topic>"): their question quoted plus the idea, text and picture ` +
              `together in ONE cell (mo.vstack + netviz).`,
          failed: false,
        });
      }
      return toResult({
        out: madeFailed
          ? `Logged. Souvenir cell FAILED — add it with nb_add_cell.`
          : `Logged and the souvenir is in their notebook.`,
        failed: false,
      });
    },
    ...quiet("Keeping a note of your question…"),
  });

  // ── nb_add_cell ───────────────────────────────────────────────────────────
  pi.registerTool({
    name: "nb_add_cell",
    label: "Add notebook cell",
    description:
      "Create and run a new cell in the live marimo notebook. " +
      MARIMO_CELL_RULES,
    promptSnippet: "Add and run a cell in the live marimo notebook",
    promptGuidelines: [
      "Use nb_add_cell / nb_edit_cell / nb_delete_cell / nb_read / nb_run for ALL notebook work — never bash, never raw marimo._code_mode boilerplate.",
      "Every nb_* status is shown to the student: short, warm, plain words only.",
    ],
    parameters: Type.Object({
      status: STATUS_PARAM,
      name: Type.String({
        description: "Unique snake_case cell name, e.g. 'cp2_ripple'. Use it later with nb_edit_cell/nb_delete_cell.",
      }),
      code: Type.String({ description: "The cell body (Python)." }),
      show_code: Type.Optional(
        Type.Boolean({
          description: "Show the code editor to the student (default false). Use true for cells whose code the student should read.",
        }),
      ),
    }),
    async execute(_id, params, signal) {
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);
      await flushParkedNotes(signal);
      await ensureChapterHeader(signal);
      const hide = params.show_code === true ? "False" : "True";
      // nb_add_exercise and nb_view_image both validate the name; this one did
      // not, and it is the tool that gets hand-written names — detours,
      // souvenirs, the personalised half of the notebook. marimo stores a cell
      // whose name is not an identifier as an unparsable block, so "detour:
      // clustering" becomes a dead region in the keepsake and neither
      // nb_edit_cell nor nb_delete_cell can address it afterwards. Correct it
      // rather than refuse, and say what it became.
      const wanted = String(params.name ?? "").trim();
      const name = /^[A-Za-z_]\w*$/.test(wanted)
        ? wanted
        : sanitize(wanted)
            .replace(/^(?=\d)/, "c_")
            .replace(/^_+$/, "cell") || "cell";
      let inner =
        `async with cm.get_context() as ctx:\n` +
        `    _cid = ctx.create_cell(_code, name=${py(name)}, hide_code=${hide})\n` +
        `    ctx.run_cell(_cid)\n`;
      inner += focusCellCode("_cid", "    ");
      // Improvised cells go through the review (nb_review.py) — it catches the
      // displays marimo would silently drop before the student sees a cell
      // with a missing figure.
      const review = reviewSource();
      // Two refusals per cell name, then it goes in with the complaint
      // attached — the same backstop every other guard here has. A review
      // the model cannot satisfy must never be able to strand the student
      // in a retry loop; a cell with one bad formula is the smaller harm,
      // and nb_edit_cell can still fix it.
      const strikes = cellReviewWarned.get(name) ?? 0;
      const enforce = review && strikes < 2;
      if (review && !enforce) cellReviewWarned.delete(name);
      else if (review) cellReviewWarned.set(name, strikes + 1);
      const code =
        `import marimo._code_mode as cm\n` +
        (review ? review + "\n" : "") +
        `_code = ${py(stripRedundantImports(params.code))}\n` +
        (review
          ? `_code, _note, _fatal = _nb_review(_code)\n` +
            (enforce
              ? `if _fatal:\n` +
                `    print(_fatal)\n` +
                `else:\n` +
                indentBlock(inner, 4) +
                `    if _note:\n` +
                `        print(_note)\n`
              : indentBlock(inner, 0) +
                `if _fatal or _note:\n` +
                `    print("INSERTED ANYWAY (third attempt) —", _fatal or _note, ` +
                `"Fix it with nb_edit_cell rather than retrying.")\n`)
          : inner);
      const addCellResult = await runKernel(code, signal);
      if (!addCellResult.failed) {
        await pinAppealToBottom(signal);
        if (name !== wanted) {
          addCellResult.out +=
            `\nNAMED "${name}" — "${wanted}" is not a usable cell name (marimo needs a ` +
            `Python identifier). Use "${name}" if you refer to this cell later.`;
        }
      }
      return toResult(addCellResult);
    },
    ...quiet("Adding something to your whiteboard…"),
  });

  // ── nb_add_exercise ───────────────────────────────────────────────────────
  // Fill-in coding, app-view friendly: instructions + a pre-filled code box
  // (mo.ui.code_editor) + a ▶ Run button that executes via the notebook's
  // run_student_code helper (stdout + last expression, friendly errors).
  // The student never needs the cell editor.
  pi.registerTool({
    name: "nb_add_exercise",
    label: "Add coding exercise",
    description:
      "Give the student a fill-in coding exercise INSIDE the notebook page: instructions, " +
      "a code box pre-filled with your scaffold (numbered # steps with ... blanks), and a " +
      "▶ Run button that executes it and shows output or a friendly error. They can run as " +
      "often as they like. Read their attempt with nb_read('<name>_ed.value'). env_vars " +
      "is ONLY for variables another notebook cell already defines (a graph G you built " +
      "earlier) — never the scaffold's own variables, which live inside the code box. " +
      "lists notebook variables their code may use (e.g. a graph G you set up earlier). " +
      "ALWAYS use this instead of asking the student to edit cells. Pass checkpoint when this " +
      "exercise IS a checkpoint's build (not a detour) — lets the tool catch a checkpoint you " +
      "started but never closed with checkpoint_done.",
    promptSnippet: "Insert a fill-in coding exercise (code box + Run button) into the notebook",
    parameters: Type.Object({
      status: STATUS_PARAM,
      name: Type.String({ description: "Base name, e.g. 'cs1_code'." }),
      instructions: Type.String({
        description: "1-3 sentences shown above the code box (markdown, $math$ ok).",
      }),
      scaffold: Type.String({
        description: "Pre-filled Python: numbered # instructions + ... blanks to fill.",
      }),
      env_vars: Type.Optional(
        Type.Array(Type.String(), {
          description: "Notebook variable names the student's code may use.",
        }),
      ),
      checkpoint: Type.Optional(
        Type.String({
          description: "Checkpoint id this build is for, e.g. 'cp6_large_n_experiment'. Omit for detours.",
        }),
      ),
    }),
    async execute(_id, params, signal) {
      const name = String(params.name ?? "").trim();
      if (!/^[A-Za-z_]\w*$/.test(name)) {
        return toResult({ out: `'${name}' is not a valid cell name.`, failed: true });
      }
      const exCpId = String(params.checkpoint ?? "").trim();
      if (exCpId && paceUnasked.has(exCpId)) {
        paceUnasked.delete(exCpId);
        return toResult({
          out:
            `NOT INSERTED — the "where to next?" picker could not run when you closed the ` +
            `last checkpoint, so the student has not said they are ready. Ask them in ` +
            `plain text, END YOUR TURN, and wait. When they say yes, call this again.`,
          failed: false,
        });
      }
      const exKey = `exercise:${exCpId}`;
      if (
        exCpId &&
        pendingCheckpoint &&
        isAheadOf(exCpId, pendingCheckpoint) &&
        (buildOrderWarned.get(exKey) ?? 0) < 2
      ) {
        buildOrderWarned.set(exKey, (buildOrderWarned.get(exKey) ?? 0) + 1);
        return toResult({
          out:
            `NOT INSERTED — '${exCpId}' comes after '${pendingCheckpoint}', which is still ` +
            `open. Call checkpoint_done for '${pendingCheckpoint}' first (its note cell ` +
            `must land before this build), then retry this insert.`,
          failed: false,
        });
      }
      // env_vars becomes a REAL reference in the generated cell, which is
      // how marimo passes a notebook variable into the exercise — and how a
      // name that does not exist takes the whole notebook down. A live run
      // listed the scaffold's OWN variables (N, k, p_values, L0, C0, rows,
      // df), which live only inside the editor's text, and the export died
      // with "name 'p_values' is not defined" on a cell the student cannot
      // even see. So only names the kernel actually defines get through.
      const asked = (params.env_vars ?? []).filter((v: string) => /^[A-Za-z_]\w*$/.test(v));
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);
      let envVars: string[] = [];
      let droppedEnv = asked;
      if (asked.length) {
        // AFTER ensureWarm: on a cold kernel every name reads as undefined,
        // and a resume that lands straight on the coding checkpoint makes
        // this the session's first nb_* call. And it fails CLOSED — letting
        // an unchecked name through is what broke `marimo export` on a cell
        // the student cannot see, while dropping one costs nothing here
        // (run_student_code already injects mo/ig/nx/np/plt/alt/pd/netviz).
        const probe = await runKernel(
          `print("DEFINED<<" + ",".join(_n for _n in ${pyList(asked)} if _n in globals()) + ">>")\n`,
          signal,
        );
        const m = /DEFINED<<(.*)>>/.exec(probe.out ?? "");
        if (!probe.failed && m) {
          const defined = new Set(m[1].split(",").filter(Boolean));
          envVars = asked.filter((v) => defined.has(v));
          droppedEnv = asked.filter((v) => !defined.has(v));
        }
      }
      const envDict = `{${envVars.map((v: string) => `${py(v)}: ${v}`).join(", ")}}`;
      await flushParkedNotes(signal);
      await ensureChapterHeader(signal);
      const edBody =
        `${name}_ed = mo.ui.code_editor(value=${py(params.scaffold)}, language="python", min_height=140)\n` +
        `${name}_run = mo.ui.run_button(label="▶ Run my code")\n` +
        `mo.vstack([mo.md(${pyMd(params.instructions)}), ${name}_ed, ${name}_run])`;
      // Every ▶ Run writes the code to assets/exercises/<name>.py. marimo
      // does NOT serialise a code_editor's value, so without this the one
      // coding checkpoint left nothing in the submitted notebook: reopen it
      // and the box is back to the scaffold, the chart is gone, and the
      // grader sees `...` blanks where the student's work was. Same cure as
      // the photo view cell — save the artifact, render it from disk.
      const savedPath = `assets/exercises/${name}.py`;
      const outBody =
        `from pathlib import Path as _P\n` +
        `_saved = _P(${py(savedPath)})\n` +
        `${name}_send = mo.ui.run_button(\n` +
        `    label="📨 Send my code to my tutor",\n` +
        `    disabled=not (${name}_run.value or _saved.exists()),\n` +
        `)\n` +
        `if ${name}_run.value:\n` +
        `    _saved.parent.mkdir(parents=True, exist_ok=True)\n` +
        `    _saved.write_text(${name}_ed.value)\n` +
        `    _res = mo.vstack([\n` +
        `        run_student_code(${name}_ed.value, ${envDict}),\n` +
        `        mo.md(\n` +
        `            "<span style='color:#6A6D75;font-size:13px'>Run it as many times "\n` +
        `            "as you like. When it does what you want, press 📨 — that is what "\n` +
        `            "hands it in and tells your tutor to look.</span>"\n` +
        `        ),\n` +
        `        ${name}_send,\n` +
        `    ])\n` +
        `elif _saved.exists():\n` +
        `    _res = mo.vstack([\n` +
        `        mo.md(\n` +
        `            "<span style='color:#6A6D75;font-size:13px'>The code I wrote and "\n` +
        `            "ran — press ▶ Run again after an edit to refresh this:</span>"\n` +
        `            "\\n\\n\`\`\`python\\n"\n` +
        `            + _saved.read_text()\n` +
        `            + "\\n\`\`\`"\n` +
        `        ),\n` +
        `        run_student_code(_saved.read_text(), ${envDict}),\n` +
        `        ${name}_send,\n` +
        `    ])\n` +
        `else:\n` +
        `    _res = mo.md("*Press ▶ Run when you're ready.*")\n` +
        `_res`;
      const sentBody =
        `from pathlib import Path as _P\n` +
        `if ${name}_send.value:\n` +
        `\n` +
        `    _P("session_artifacts").mkdir(exist_ok=True)\n` +
        `    with open("session_artifacts/student_signal.txt", "a") as _f:\n` +
        `        _f.write(${py(name + "_ed")} + "\\n")\n` +
        `    _sent = mo.md("✅ **Handed in.** Your tutor is reading your code now.")\n` +
        `else:\n` +
        // Not mo.md(""): an empty markdown node is a blank cell in the
        // keepsake, and reopening it always lands on this branch.
        // Only once the button is actually on screen — before the first
        // ▶ Run the out cell renders "Press ▶ Run when you're ready" and no
        // 📨 button at all, and pointing at it then is one more thing for a
        // nervous beginner to hunt for.
        // Phrased so it is still true months later: "press 📨 once the chart
        // looks right" is an instruction to a session that ended, in a
        // notebook whose code was handed in long ago. Naming what the button
        // does works live AND on a cold read.
        `    _sent = mo.md(\n` +
        `        "<span style='color:#6A6D75;font-size:13px'>*The 📨 button above is "\n` +
        `        "what hands this code to your tutor.*</span>"\n` +
        `        if _P(${py(savedPath)}).exists()\n` +
        `        else "<span style='color:#6A6D75;font-size:13px'>*A 📨 hand-in button "\n` +
        `        "appears here once you have pressed ▶ Run.*</span>"\n` +
        `    )\n` +
        `_sent`;
      let code =
        `import marimo._code_mode as cm\n` +
        `from pathlib import Path as _P\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _names = [c.name for c in ctx.cells]\n` +
        `    if ${py(name + "_ed")} in _names:\n` +
        `        print("exercise already in the notebook — skipped duplicate insert")\n` +
        `    else:\n` +
        // Handing out the box means starting it. Any saved file here belongs
        // to someone else's session (or, once, to a solved copy committed by
        // mistake), and the out cell renders whatever it finds under "The
        // code I wrote and ran" — which would show this student the answer
        // before they typed a character, in their own voice.
        `        _P(${py(savedPath)}).unlink(missing_ok=True)\n` +
        `        _cid = ctx.create_cell(${py(edBody)}, name=${py(name + "_ed")}, hide_code=True)\n` +
        `        ctx.run_cell(_cid)\n` +
        `        _first = _cid\n` +
        `        _cid = ctx.create_cell(${py(outBody)}, name=${py(name + "_out")}, hide_code=True, after=_cid)\n` +
        `        ctx.run_cell(_cid)\n` +
        `        _cid = ctx.create_cell(${py(sentBody)}, name=${py(name + "_sent")}, hide_code=True, after=_cid)\n` +
        `        ctx.run_cell(_cid)\n`;
      code += focusCellCode("_first", "        ");
      const result = await runKernel(code, signal);
      if (!result.failed) await pinAppealToBottom(signal);
      if (!result.failed) {
        result.out =
          `Exercise inserted. The student sees your instructions, a runnable code box, a ` +
          `▶ Run button, and — once they have run it — a 📨 Send button that hands the ` +
          `code in. Ask for the code, then WAIT: their press starts your turn. Every run ` +
          `saves the code to ${savedPath}, so it is still in the notebook months later.\n` +
          (droppedEnv.length
            ? `(Ignored env_vars this notebook does not define: ${droppedEnv.join(", ")} — ` +
              `those look like your scaffold's own variables, which live inside the code ` +
              `box. Nothing to fix unless you meant a variable an earlier cell made.)\n`
            : "") +
          result.out;
      }
      return toResult(result);
    },
    ...quiet("Setting up a box for you to try…"),
  });

  // ── nb_add_template ───────────────────────────────────────────────────────
  // Premade, tested cell groups shipped in cells/*.py — the model sends only
  // a template name, so scripted checkpoint builds are instant and bug-free.
  pi.registerTool({
    name: "nb_add_template",
    label: "Insert premade cells",
    description: (() => {
      const dir = path.join(process.cwd(), "cells");
      const names = fs.existsSync(dir)
        ? fs
            .readdirSync(dir)
            .filter((f) => f.endsWith(".py"))
            .map((f) => f.slice(0, -3))
        : [];
      return (
        "Insert a PREMADE, tested group of cells into the notebook instantly — no code to " +
        "write. ALWAYS prefer this over nb_add_cell when a template exists for the " +
        "checkpoint. Available templates: " +
        (names.join(", ") || "(none found)") +
        ". REFUSES to insert if an earlier checkpoint was started but never closed with " +
        "checkpoint_done — close it first, its note cell must land before this build."
      );
    })(),
    promptSnippet: "Insert premade, tested notebook cells by template name (instant)",
    promptGuidelines: [
      "For checkpoint builds use nb_add_template with the template named in lesson.yaml; nb_add_cell is only for detours and improvised cells.",
      "Always pass checkpoint — the id of the checkpoint this build is for, from the script.",
    ],
    parameters: Type.Object({
      status: STATUS_PARAM,
      template: Type.String({ description: "Template name, e.g. 'cp2_ripple'." }),
      checkpoint: Type.String({
        description:
          "Checkpoint id from the script this build is for, e.g. 'cp2_distance'. Required — " +
          "lets the tool catch a checkpoint you started but never closed with checkpoint_done.",
      }),
    }),
    async execute(_id, params, signal) {
      const cpId = String(params.checkpoint ?? "").trim();
      if (!cpId) {
        return toResult({
          out: `NOT INSERTED — pass checkpoint: the id of the checkpoint this build is for.`,
          failed: false,
        });
      }
      // Keyed by TOOL as well as checkpoint: a shared budget meant the
      // exercise tool's refusals spent the template tool's, and a genuine
      // first-try violation was then waved through.
      if (paceUnasked.has(cpId)) {
        paceUnasked.delete(cpId);
        return toResult({
          out:
            `NOT INSERTED — the "where to next?" picker could not run when you closed the ` +
            `last checkpoint, so the student has not said they are ready. Ask them in ` +
            `plain text, END YOUR TURN, and wait. When they say yes, call this again — it ` +
            `will go in.`,
          failed: false,
        });
      }
      const tplKey = `template:${cpId}`;
      if (
        pendingCheckpoint &&
        isAheadOf(cpId, pendingCheckpoint) &&
        (buildOrderWarned.get(tplKey) ?? 0) < 2
      ) {
        buildOrderWarned.set(tplKey, (buildOrderWarned.get(tplKey) ?? 0) + 1);
        return toResult({
          out:
            `NOT INSERTED — '${cpId}' comes after '${pendingCheckpoint}', which is still ` +
            `open. Call checkpoint_done for '${pendingCheckpoint}' first (its note cell ` +
            `must land before this build), then retry this insert.`,
          failed: false,
        });
      }
      const file = path.join(process.cwd(), "cells", `${params.template}.py`);
      if (!fs.existsSync(file)) {
        return toResult({ out: `No template named '${params.template}'.`, failed: true });
      }
      const src = fs.readFileSync(file, "utf-8");
      // Factual description the tutor can safely echo — prevents the model
      // from misdescribing the artifact (e.g. calling a 4-person network
      // "5-person", seen in production).
      const describe = /^# describe: (.+)$/m.exec(src)?.[1] ?? "";
      const parts = src.split(/^# --- cell: (\w+) ---[ \t]*$/m);
      const cells: Array<{ name: string; code: string }> = [];
      for (let i = 1; i < parts.length; i += 2) {
        cells.push({ name: parts[i], code: parts[i + 1].trim() });
      }
      if (cells.length === 0) {
        return toResult({ out: `Template '${params.template}' has no cells.`, failed: true });
      }
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);
      await flushParkedNotes(signal);
      await ensureChapterHeader(signal);
      let code =
        `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _names = [c.name for c in ctx.cells]\n` +
        `    if ${py(cells[0].name)} in _names:\n` +
        `        print("template already in the notebook — skipped duplicate insert")\n` +
        `    else:\n` +
        `        _cid = ctx.create_cell(${py(cells[0].code)}, name=${py(cells[0].name)}, hide_code=True)\n` +
        `        ctx.run_cell(_cid)\n` +
        `        _first = _cid\n`;
      for (const c of cells.slice(1)) {
        code +=
          `        _cid = ctx.create_cell(${py(c.code)}, name=${py(c.name)}, hide_code=True, after=_cid)\n` +
          `        ctx.run_cell(_cid)\n`;
      }
      code += focusCellCode("_first", "        ");
      const result = await runKernel(code, signal);
      if (!result.failed) await pinAppealToBottom(signal);
      if (!result.failed && describe) {
        // Upload widgets are named per template (cp4_photo, cp2_paperwork_photo,
        // cp5_ring_paperwork_photo…). The tutor cannot know which one it just
        // inserted, and nb_view_image with the wrong name blows up in the
        // kernel — so the insert result names it.
        const uploads = cells
          .filter((c) => /\bmo\.ui\.file\s*\(/.test(c.code))
          .map((c) => c.name);
        const uploadLine = uploads.length
          ? `ASK FOR THE PHOTO with the question it belongs to, in one line, and then ` +
            `WAIT — never ask whether it is up. Their 📨 Send to my tutor press starts ` +
            `your turn; then call nb_view_image(widget="${uploads[0]}", …).\n` +
            // "ASK FOR THE PHOTO NOW" is what this said, and a script that puts the
            // drop area up a turn BEFORE the question could not win against it: the
            // tutor announced the pen-and-paper step, built the cell and asked the
            // question in one breath, spending the one deliberate pause the
            // checkpoint has — the beat where "this one is a drawing" is allowed to
            // land before the hard part arrives. If your script gives this build its
            // own turn, keep that turn.
            `If your CHAPTER SCRIPT gives this build a turn of its own before the ` +
            `question, KEEP IT: say that beat, build this, and stop. The photo ask ` +
            `goes with the question, on the next turn.\n` +
            `Say NOTHING about typing instead, and nothing about a camera, unless they ` +
            `have mentioned one at THIS checkpoint. A camera that was out an hour ago ` +
            `is not a fact about now: a live run opened here with "your camera was out ` +
            `before — so just tell me which two dots", to a student who had a working ` +
            `camera and had said nothing. The drawing is the point of this checkpoint.\n`
          : "";
        result.out =
          `Inserted. The student now sees: ${describe}\n` +
          `(Describe it to the student ONLY from this line — never guess counts or details.)\n` +
          uploadLine +
          result.out;
      }
      return toResult(result);
    },
    ...quiet("Putting something on your whiteboard…"),
  });

  // ── nb_fresh_start ────────────────────────────────────────────────────────
  // Conversational reset: archives the previous notebook + session log, then
  // deletes every tutor-made cell from the LIVE notebook (template cells are
  // unnamed and survive). Called when the student chooses "start fresh".
  pi.registerTool({
    name: "nb_fresh_start",
    label: "Fresh start",
    description:
      "Reset the session at the student's request: archives the previous notebook and " +
      "session log to session_artifacts/, then clears all tutor-made cells from the live " +
      "notebook. Call ONLY after the student chose to start fresh (ask_user_question).",
    promptSnippet: "Archive the previous session and clear the notebook (student chose fresh start)",
    parameters: Type.Object({
      status: STATUS_PARAM,
    }),
    async execute(_id, params, signal, _onUpdate, ctx: any) {
      const stamp = new Date()
        .toISOString()
        .replace(/[-:]/g, "")
        .replace(/\..*/, "")
        .replace("T", "-");
      const dir = path.join(process.cwd(), "session_artifacts");
      try {
        fs.mkdirSync(dir, { recursive: true });
        const nb = path.join(process.cwd(), "notebook.py");
        if (fs.existsSync(nb)) fs.copyFileSync(nb, path.join(dir, `notebook-${stamp}.py`));
        const log = path.join(dir, "session_log.jsonl");
        if (fs.existsSync(log)) fs.renameSync(log, path.join(dir, `session_log-${stamp}.jsonl`));
        // The summary belongs to the run that produced it. Left in place it
        // outlives its log and describes checkpoints the new session never did.
        const sum = path.join(dir, "session_summary.md");
        if (fs.existsSync(sum)) fs.renameSync(sum, path.join(dir, `session_summary-${stamp}.md`));
        // Delete, don't rewrite: the saved chapter belongs to the log that
        // was just archived, and anything left in that file would be read
        // back as progress this session never made.
        fs.rmSync(chapterStatePath(), { force: true });
        // The student's saved exercise code goes with it. Left behind, the
        // next run of the coding checkpoint opens showing the PREVIOUS
        // student's code under "The code I wrote and ran" — someone else's
        // work, in this student's voice.
        // Both directories, for the same reason: they are the previous
        // student's work, and the photo guard reads assets/uploads as
        // evidence that a page arrived — stale files there disarm it before
        // the new student has taken a photo.
        for (const [rel, label] of [
          ["exercises", `exercises-${stamp}`],
          ["uploads", `uploads-${stamp}`],
        ] as const) {
          const src = path.join(process.cwd(), "assets", rel);
          if (!fs.existsSync(src)) continue;
          fs.renameSync(src, path.join(dir, label));
          // The archived notebook still points at assets/uploads/<w>_view.jpg,
          // which no longer exists — so repoint the archived COPY at its own
          // stamped directory. A shared session_artifacts/assets/uploads/ was
          // tried first and was worse: the next reset overwrote it, and
          // session 1's keepsake then rendered session 2's photograph as that
          // student's own hand-worked page.
          try {
            const archived = path.join(dir, `notebook-${stamp}.py`);
            if (fs.existsSync(archived)) {
              fs.writeFileSync(
                archived,
                fs.readFileSync(archived, "utf-8").split(`assets/${rel}/`).join(`${label}/`),
              );
            }
          } catch {
            // the archive is best-effort; the live notebook is what matters
          }
        }
      } catch {
        // archiving is best-effort; clearing the notebook is what matters
      }
      // Back to chapter 1 with a fresh script in context. triggerTurn so the
      // tutor starts cp0 from the script once it arrives — without this, the
      // model improvises checkpoints from memory (seen in production).
      try {
        const chapters = loadChapters();
        if (chapters.length > 0) {
          writeChapterState(chapters[0].id);
          // Re-arm the open-checkpoint guard on cp0 — a fresh start rewinds
          // the script, so a stale pending id would refuse cp1's build.
          pendingCheckpoint = chapters[0].checkpoints[0] ?? null;
          slotDriftWarned.clear();
          detourTextOnlyWarned.clear();
          // A fresh start rewinds the record too: anything picked before it
          // (the resume dialog, a previous run's answers) must not surface
          // in the new session's first checkpoint.
          pickedAnswers.length = 0;
          pickedMark = 0;
          awaitingResumeChoice = false;
          cellReviewWarned.clear();
          chapterGateWarned.clear();
          try {
            // Archived, not deleted — its file-level twin promises "nothing is
            // ever deleted", and the rendered note text is nowhere else.
            if (fs.existsSync(parkedDir())) {
              fs.renameSync(parkedDir(), path.join(dir, `parked_notes-${stamp}`));
            }
          } catch {
            // a parked note belongs to the session just archived
          }
          buildOrderWarned.clear();
          paceUnasked.clear();
          resumeGaps.clear();
          viewedPhotos.clear();
          // Same rewind for the transcript mark: without it, whatever the
          // student typed before choosing "start fresh" is filed as the new
          // cp0's own words.
          studentSaidSince(ctx, true);
          pi.sendMessage(
            {
              customType: "chapter-script",
              content: chapterScriptMessage(chapters[0], 1, chapters.length),
              display: false,
            },
            { deliverAs: "followUp", triggerTurn: true },
          );
        }
      } catch {
        // best-effort
      }
      let code =
        `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    for _c in list(ctx.cells):\n` +
        `        if _c.name and _c.name != "_":\n` +
        `            ctx.delete_cell(_c.id)\n` +
        // Say out loud what survived. A wipe that silently leaves a cell
        // behind is how a "clean slate" notebook opens on the middle of the
        // module — the tutor must know before it starts building.
        `    _left = [c.name for c in ctx.cells if c.name and c.name != "_"]\n` +
        `    if _left:\n` +
        `        print("STILL THERE (delete failed):", ", ".join(_left))\n`;
      // Same kernel call as the wipe so the chapter header lands first,
      // before any cp0/cp1 build cells.
      try {
        const chapters = loadChapters();
        if (chapters.length > 0) {
          const h = `${chapters[0].id}_header`;
          const heading = `## Chapter 1 of ${chapters.length} — ${chapters[0].title}`;
          const op = chapterOpening(chapters[0]);
          const body = `mo.md(${pyMd(op ? `${heading}\n\n${op}` : heading)})`;
          code +=
            `    _cid = ctx.create_cell(${py(body)}, name=${py(h)}, hide_code=True)\n` +
            `    ctx.run_cell(_cid)\n`;
        }
      } catch {
        // header is cosmetic
      }
      const result = await runKernel(code, signal);
      if (!result.failed) await pinAppealToBottom(signal);
      if (!result.failed) {
        result.out =
          `Fresh start complete. The Chapter 1 script arrives next — END YOUR TURN NOW ` +
          `(at most one short welcome line first). Treat this as a brand-new session: ` +
          `begin at cp0_welcome FROM THE INCOMING SCRIPT; do not improvise checkpoints ` +
          `from memory.\n` + result.out;
      }
      return toResult(result);
    },
    ...quiet("Clearing the whiteboard for a fresh start…"),
  });

  // ── nb_edit_cell ──────────────────────────────────────────────────────────
  pi.registerTool({
    name: "nb_edit_cell",
    label: "Edit notebook cell",
    description:
      "Replace the body of an existing notebook cell (by the name you gave it in nb_add_cell) and re-run it. " +
      "Submit the FULL new body. " + MARIMO_CELL_RULES,
    promptSnippet: "Edit and re-run a cell in the live marimo notebook",
    parameters: Type.Object({
      status: STATUS_PARAM,
      name: Type.String({ description: "The cell's name." }),
      code: Type.String({ description: "The full replacement cell body (Python)." }),
    }),
    async execute(_id, params, signal) {
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);
      const code =
        `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _names = [c.name for c in ctx.cells]\n` +
        `    if ${py(params.name)} not in _names:\n` +
        `        print("EDIT FAILED: no cell named", ${py(params.name)})\n` +
        `        print("Existing cells:", [n for n in _names if n and n != "_"])\n` +
        `    else:\n` +
        `        ctx.edit_cell(${py(params.name)}, ${py(stripRedundantImports(params.code))})\n` +
        `        ctx.run_cell(${py(params.name)})\n`;
      return toResult(await runKernel(code, signal));
    },
    ...quiet("Tidying something on your whiteboard…"),
  });

  // ── nb_delete_cell ────────────────────────────────────────────────────────
  pi.registerTool({
    name: "nb_delete_cell",
    label: "Delete notebook cells",
    description:
      "Delete notebook cells by name. Deleting removes the variables those cells define. " +
      "Never delete a cell holding a student's answer.",
    promptSnippet: "Delete cells from the live marimo notebook",
    parameters: Type.Object({
      status: STATUS_PARAM,
      names: Type.Array(Type.String(), { description: "Cell names to delete." }),
    }),
    async execute(_id, params, signal) {
      // "Never delete a cell holding a student's answer" lived only in the
      // description above, and a model whose cell came out wrong reaches for
      // delete-and-rebuild by reflex. The cells that reflex reaches for are
      // the ones that cannot be rebuilt: a note cell quotes words that, after
      // compaction, are no longer anywhere in the model's context; a _view
      // cell is the only copy of the photograph inside the notebook; an _ed
      // cell is the code the student wrote. All three are the graded artifact.
      // _preview holds no student work, which is how it got left off the list
      // — but it is where the 📨 button is DEFINED, and _sent reads it, so
      // deleting it leaves a protected cell referencing an unbound name and
      // the student with no way to hand the photo in. Protect it too: the
      // partial path below deletes what it is allowed to and reports success,
      // so "only the unprotected one went" is not a safe failure.
      const names: string[] = Array.isArray(params.names) ? params.names.map(String) : [];
      const PROTECTED = /(_note|_view|_photo|_preview|_ed|_out|_sent|_header)$|^session_record$/;
      const refused = names.filter((n) => PROTECTED.test(n));
      const allowed = names.filter((n) => !PROTECTED.test(n));
      if (refused.length && !allowed.length) {
        return toResult({
          failed: false,
          out:
            `NOT DELETED — ${refused.map((n) => `"${n}"`).join(", ")} ${refused.length > 1 ? "hold" : "holds"} ` +
            `the student's own work (their answer, their photo, or their code) and ` +
            `${refused.length > 1 ? "are" : "is"} part of what gets graded. Nothing was ` +
            `deleted. If the content is wrong, fix it with nb_edit_cell — and if it is a ` +
            `note cell, leave it alone: checkpoint_done wrote it from the transcript.`,
        });
      }
      const code =
        `import marimo._code_mode as cm\n` +
        `async with cm.get_context() as ctx:\n` +
        `    _names = [c.name for c in ctx.cells]\n` +
        `    for _n in ${pyList(allowed)}:\n` +
        `        if _n in _names:\n` +
        `            ctx.delete_cell(_n)\n` +
        `        else:\n` +
        `            print("skip: no cell named", _n)\n`;
      const r = await runKernel(code, signal);
      if (refused.length && !r.failed) {
        r.out +=
          `\nKEPT: ${refused.map((n) => `"${n}"`).join(", ")} — the student's own work is ` +
          `never deleted. Use nb_edit_cell if one of them needs a correction.`;
      }
      return toResult(r);
    },
    ...quiet("Clearing something off the whiteboard…"),
  });

  // ── nb_read ───────────────────────────────────────────────────────────────
  pi.registerTool({
    name: "nb_read",
    label: "Read notebook values",
    description:
      "Evaluate expressions against the live notebook and return their values — the way to read " +
      "student widget answers, e.g. ['cp6_p.value', 'cp4_photo.value[0].name']. Returns one line per expression.",
    promptSnippet: "Read values/widget answers from the live marimo notebook",
    parameters: Type.Object({
      status: STATUS_PARAM,
      expressions: Type.Array(Type.String(), { description: "Python expressions to evaluate." }),
    }),
    async execute(_id, params, signal) {
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);
      const code =
        `for _e in ${pyList(params.expressions)}:\n` +
        `    try:\n` +
        `        print(_e, "=", repr(eval(_e)))\n` +
        `    except Exception as _ex:\n` +
        `        print(_e, "!", type(_ex).__name__, str(_ex))\n`;
      return toResult(await runKernel(code, signal));
    },
    ...quiet("Having a look at your notebook…"),
  });

  // ── nb_view_image ─────────────────────────────────────────────────────────
  // The tutor model is text-only; this tool is its eyes. Kernel side: pull the
  // upload bytes, save the original (graded artifact), EXIF-rotate + downscale
  // (phone photos are huge and often sideways), show the photo back in the
  // notebook. Extension side: send the small JPEG to a vision model and hand
  // the tutor a factual description to judge.
  pi.registerTool({
    name: "nb_view_image",
    label: "View student image",
    description:
      "Look at a student-uploaded image — you are text-only, so this is your ONLY way to " +
      "see one (never nb_read image bytes). Give the upload widget name (e.g. 'cp4_photo') " +
      "or a file path, what the task was, and the question you need answered. It saves the " +
      "original to assets/uploads/ so it travels with the notebook, shows the photo in the " +
      "notebook for the student, and " +
      "returns a factual description from a vision model. The description is a machine's " +
      "reading, not ground truth — confirm the key detail with the student before building " +
      "on it. If it reports no vision is available, follow its advice instead.",
    promptSnippet: "See a student-uploaded image through a vision model (the tutor is text-only)",
    parameters: Type.Object({
      status: STATUS_PARAM,
      widget: Type.Optional(
        Type.String({ description: "Upload widget name, e.g. 'cp4_photo'." }),
      ),
      file: Type.Optional(Type.String({ description: "Or: path to an image file." })),
      task: Type.String({
        description:
          "What the student was asked to draw/do, in 1-2 sentences copied from the " +
          "checkpoint — the vision model needs this to know what to look for.",
      }),
      question: Type.String({
        description: "What you need to know, e.g. 'Which two dots does the extra line connect?'",
      }),
    }),
    async execute(_id, params, signal, _onUpdate, ctx: any) {
      const widget = String(params.widget ?? "").trim();
      const file = String(params.file ?? "").trim();
      if (!widget && !file) {
        return toResult({ out: "Pass widget (e.g. 'cp4_photo') or file.", failed: true });
      }
      if (widget && !/^[A-Za-z_]\w*$/.test(widget)) {
        return toResult({ out: `'${widget}' is not a widget name.`, failed: true });
      }
      const warm = await ensureWarm(signal);
      if (warm) return toResult(warm);

      const base = sanitize(widget || path.basename(file).replace(/\.[^.]*$/, ""));
      // assets/, not session_artifacts/: the photo is the student's own figure
      // in the keepsake notebook, so it has to travel WITH the module. The
      // archive folder is gitignored evidence and may not be carried along —
      // a notebook pointing there throws FileNotFoundError on reopen.
      const viewRel = `assets/uploads/${base}_view.jpg`;
      const viewCell = `${base}_view`;
      // Self-contained display cell: the student sees exactly what the vision
      // model was sent (survives notebook reloads; deleted by fresh_start).
      // The caption is not decoration — an uncaptioned phone photo in the
      // middle of a lecture note is a mystery to anyone reading it cold,
      // including the student in three months. `task` already holds the ask
      // in the tutor's own words, so the cell states what the page shows.
      const taskLine = String(params.task ?? "")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/["']/g, "");
      const caption =
        `<span style='color:#6A6D75;font-size:13px'>📷 My own work on paper` +
        (taskLine ? ` — the task: ${taskLine}` : "") +
        `</span>`;
      const cellBody =
        `from pathlib import Path as _P\n` +
        `mo.vstack([\n` +
        `    mo.image(_P(${py(viewRel)}).read_bytes(), width=420),\n` +
        `    mo.md(${pyMd(caption)}),\n` +
        `])`;
      const source = widget
        ? `_files = list(${widget}.value or [])\n` +
          `if not _files:\n` +
          `    print("NO_IMAGE: nothing uploaded yet")\n` +
          `else:\n` +
          `    _name, _raw = _files[0].name, _files[0].contents\n`
        : `_p = _P(${py(file)})\n` +
          `if not _p.exists():\n` +
          `    print("NO_IMAGE: no such file:", ${py(file)})\n` +
          `else:\n` +
          `    _name, _raw = _p.name, _p.read_bytes()\n`;
      const code =
        `import base64 as _b64, io as _io\n` +
        `from pathlib import Path as _P\n` +
        `_P("assets/uploads").mkdir(parents=True, exist_ok=True)\n` +
        `_raw = None\n` +
        source +
        `if _raw is not None:\n` +
        `    from PIL import Image as _Image, ImageOps as _ImageOps\n` +
        `    _ext = _P(_name).suffix.lower() or ".png"\n` +
        `    (_P("assets/uploads") / (${py(base + "_upload")} + _ext)).write_bytes(_raw)\n` +
        `    _img = _Image.open(_io.BytesIO(_raw))\n` +
        `    _img = (_ImageOps.exif_transpose(_img) or _img).convert("RGB")\n` +
        `    _img.thumbnail((1280, 1280))\n` +
        `    _out = _io.BytesIO()\n` +
        `    _img.save(_out, "JPEG", quality=80)\n` +
        `    _P(${py(viewRel)}).write_bytes(_out.getvalue())\n` +
        `    print("FILE:", _name, "->", str(_img.size[0]) + "x" + str(_img.size[1]))\n` +
        `    import marimo._code_mode as cm\n` +
        `    async with cm.get_context() as ctx:\n` +
        `        _names = [c.name for c in ctx.cells]\n` +
        // The upload box's live preview reads mo.ui.file.value, which marimo
        // does NOT serialise — reopen the keepsake and the student's own
        // hand-worked page is gone, replaced by "your photo appears here".
        // So the saved file gets its own cell after all. The two do not
        // disagree after a redo: this cell is EDITED on every nb_view_image
        // call, so the preview shows what is in the box and this shows the
        // page the tutor actually read and graded.
        `        if ${py(viewCell)} in _names:\n` +
        `            ctx.edit_cell(${py(viewCell)}, ${py(cellBody)})\n` +
        `            ctx.run_cell(${py(viewCell)})\n` +
        focusCellCode(`ctx.cells[${py(viewCell)}].id`, "            ") +
        `        else:\n` +
        `            ctx.create_cell(${py(cellBody)}, name=${py(viewCell)}, hide_code=True)\n` +
        `            ctx.run_cell(${py(viewCell)})\n` +
        focusCellCode(`ctx.cells[${py(viewCell)}].id`, "            ") +
        `    print("B64:" + _b64.b64encode(_out.getvalue()).decode())\n`;
      const result = await runKernel(code, signal);
      if (result.failed) return toResult(result);
      await pinAppealToBottom(signal);
      if (result.out.includes("NO_IMAGE")) {
        return toResult({
          out:
            result.out +
            `\nThe drop box is EMPTY — they pressed Send before the photo attached, so ` +
            `they are waiting on you and do not know why. Say so warmly in one line ` +
            `("the box came through empty — drop the photo in and press send again"), ` +
            `then end your turn. Never answer this one with silence: they just pressed ` +
            `a button and a quiet screen reads as broken.`,
          failed: false,
        });
      }
      const b64 = /^B64:([A-Za-z0-9+/=]+)\s*$/m.exec(result.out)?.[1];
      const fileLine = /^FILE:.*$/m.exec(result.out)?.[0] ?? "";
      // A photo actually reached the tutor for this widget — the one piece
      // of evidence checkpoint_done can use to tell "asked for the page" from
      // "never mentioned it".
      if (widget) viewedPhotos.add(widget);
      if (!b64) {
        return toResult({
          out:
            `Could not extract the image. Ask the student to describe their drawing ` +
            `in words instead and judge that.\n${result.out.slice(0, 500)}`,
          failed: true,
        });
      }
      const vision = await describeImage(
        ctx,
        b64,
        String(params.task ?? "").trim(),
        String(params.question ?? "").trim(),
      );
      if (vision.failed) return toResult({ out: `${fileLine}\n${vision.text}`, failed: false });
      return toResult({
        out:
          `${fileLine} — saved, and the student now sees their photo in the notebook.\n` +
          `VISION REPORT from ${vision.model} (a machine description — judge it against ` +
          `the checkpoint yourself; respond to a concrete detail from it, don't echo it ` +
          `wholesale):\n${vision.text}`,
        failed: false,
      });
    },
    ...quiet("Looking at your photo…"),
  });

  // ── nb_run ────────────────────────────────────────────────────────────────
  pi.registerTool({
    name: "nb_run",
    label: "Run Python in notebook kernel",
    description:
      "Escape hatch: run arbitrary Python in the notebook's scratchpad (variables visible, new " +
      "top-level bindings discarded). Use for: appending to the session log, saving uploaded " +
      "photo bytes to session_artifacts/, timestamps (datetime), quick computations. " +
      "NOT for creating/editing cells — use nb_add_cell/nb_edit_cell for that. " +
      "AND NEVER TO PRE-SOLVE THE OPEN CHECKPOINT: a live run worked out a checkpoint's " +
      "own arithmetic here and said the answer in the next breath, so the student never " +
      "answered the question they were asked. If a number is what the checkpoint is " +
      "asking for, you may not be the one who says it — ask, end your turn, wait.",
    promptSnippet: "Run scratchpad Python in the notebook kernel (logging, file saves, checks)",
    parameters: Type.Object({
      status: STATUS_PARAM,
      code: Type.String({ description: "Python code to run in the scratchpad." }),
    }),
    async execute(_id, params, signal) {
      const result = await runKernel(params.code, signal);
      // Hand-written log JSON is obsolete and drifts (schema, timestamps,
      // paraphrased answers) — redirect to the tool that owns the record.
      if (/session_log|session_summary/.test(params.code)) {
        result.out =
          (result.out ? result.out + "\n" : "") +
          `NOTE: do NOT write the session log or summary by hand — checkpoint_done ` +
          `logs checkpoints (and adds the note cell and the transition ask), ` +
          `log_detour logs questions, and chapter_done writes the closing summary.`;
      }
      return toResult(result);
    },
    ...quiet("One moment…"),
  });

  // ── waiting-time trivia ───────────────────────────────────────────────────
  // The student stares at "Working…" during model calls, notebook round-trips
  // and chapter handoffs. Replace it with a rotating network-science tidbit —
  // dead air becomes a tiny extra lesson.
  //
  // HARD RULE — these lines are shown at random, so any one of them can land
  // while any checkpoint is open, in ANY module that installs this toolkit.
  // One list serves them all, so it must be clear of every module's answers.
  //
  // Small-world (m02): no hop count for Milgram or Facebook (cp1), no "put
  // the cable across the ring" (cp4), no friendship/triangle counts (cp3,
  // cp5), no "clustering stays while distance collapses" mechanism (cp6), and
  // no "high clustering AND short paths" definition (cp5_tension, cp7).
  //
  // Königsberg (m01): no degree counts and no "two ends per edge, so the
  // total is even" (cp2_degree), nothing about distances or sizes being
  // irrelevant (cp2_abstraction), no odd-node rule and no 0-or-2 condition
  // (cp3_parity, cp3_verdict), nothing about what the 1944 raid did to the
  // answer (cp3_bombing), no connectivity clause (cp4_disconnected,
  // cp6_redteam), and no indptr rule (cp5_csr). This module is the reason
  // three of the lines below were rewritten: "geometry with the distances
  // thrown away" is cp2_abstraction's gold answer in six words, "the puzzle's
  // answer changed with them" is cp3_bombing's, and the handshake fact is the
  // hint cp2_degree's teaching moment turns on. Context and history are fine;
  // answers are not.
  //
  // And a line that says "this module" must be true in every module it can
  // appear in. One claimed Karinthy asked the question behind "this whole
  // module" — true next door, and simply false to a student reading about
  // Euler.
  //
  // A NAME is an answer too. cp3_global_clustering asks the student what to
  // call a trio with all three friendships present, and a tip reading "that's
  // where triangles come from" handed the word over while the question was
  // open — it cleared the count rule above and still lost the checkpoint. If
  // a checkpoint asks "what would you call…", the word it wants is banned
  // here as firmly as any number.
  // Scoped by module. One flat list served every module, and an m01 student
  // counting bridges was told "that 1998 Nature paper is three pages long" —
  // about a paper nothing in their session had mentioned — along with karate
  // clubs, Milgram's letters and C. elegans. None of it leaked an answer; all
  // of it was orphaned, and half of it primes the NEXT module's subject in a
  // course whose own rule is to point backward and never forward. Two live
  // probes filed it independently.
  //
  // GENERAL is what survives anywhere: it may not depend on anything a
  // particular module introduced, and it may not use "that"/"his"/"the famous
  // result" to point at something the student has not met.
  const TRIVIA_GENERAL = [
    "Leonhard Euler wrote roughly 850 papers — and kept publishing after going blind.",
    "Cayley counted trees in 1889 for the chemists, who wanted to know how many molecules were possible.",
    "The word 'graph' for a network comes from Sylvester, 1878 — borrowed from chemistry diagrams.",
    "The four-colour theorem, 1976, was the first famous proof a computer helped finish. It's about a graph.",
    "Graph theory waited two centuries for its first textbook: Dénes König's, in 1936.",
    "Graph isomorphism — are these two drawings the same network? — still has no known fast algorithm.",
    "Thirty people allow more possible friendship patterns than the universe has atoms.",
    "A fruit fly's whole brain was mapped in 2024: about 140,000 neurons, 50 million connections.",
    "Complete wiring maps exist for a worm, a fly, and one cubic millimetre of mouse cortex. That's the list.",
    "Neuroscience calls the brain's wiring diagram a connectome — network science, with wetware.",
    // — the karate club —
    "Zachary's karate club split in two in 1977 — and became network science's favorite dataset.",
    "Zachary sat and watched that club argue for two years. The dataset is field notes.",
    "Zachary's own method called the split almost perfectly: one member ended up on the wrong side.",
    "Find the karate-club split at a network conference and you can win an actual trophy.",
    // — Erdős, Bacon, and counting your distance to people —
    "Erdős number: your coauthor distance to Paul Erdős, who has number 0 and 500-odd coauthors.",
    "About 11,000 people have Erdős number 2, and most of them never met him.",
    "Paul Erdős owned almost nothing and lived out of a suitcase, in other mathematicians' spare rooms.",
    "In Erdős's private vocabulary, a mathematician who stopped working had 'died'; one who died had 'left'.",
    "Erdős and Rényi's 1959 random graph — throw every link in at random — is still the field's null model.",
    "Erdős number + Bacon number = the Erdős–Bacon number. Natalie Portman has one.",
    "Kevin Bacon isn't Hollywood's center — hundreds of actors are better connected.",
    "Three college students invented the Kevin Bacon game in 1994, after watching Footloose.",
    "The Oracle of Bacon has been computing actor distances on the web since 1996.",
    // — how sociologists got here —
    "Granovetter, 1973: people find jobs through acquaintances, not close friends. Weak ties win.",
    "Granovetter's weak-ties paper was rejected in 1969. It is now one of the most cited in all of sociology.",
    "Simmel, 1908: add one person to a pair and you get politics — alliances, mediators, majorities.",
    "Moreno drew the first social network by hand in 1933; the New York Times covered it.",
    "Moreno's first sociograms were drawn to explain a wave of runaways at a girls' reform school.",
    "Network scientists call a network's own map of itself a 'sociogram' — Moreno's word, still in use.",
    "Bavelas wired lab groups as a circle, a chain or a star in the 1940s, then timed them solving puzzles.",
    "Freeman formalised betweenness in 1977: how often you sit on other people's routes.",
    "Anatol Rapoport was modelling how rumours travel along acquaintance chains in the 1950s.",
    "Your friends have more friends than you do, on average — the friendship paradox.",
    "Dunbar's number: human brains manage roughly 150 stable relationships.",
    "You personally know about 0.000002% of the people alive today.",
    // — networks that run the modern world —
    "The web, citation networks, and Hollywood all share one shape: a few superstar hubs.",
    "Hub networks shrug off random failures — but fall fast to targeted attacks on hubs.",
    "Barabási and Albert, 1999: hubs come from growth plus the rich getting richer.",
    "The rich-get-richer idea is older than the web: Yule in 1925, Simon in 1955, Price in 1965.",
    "Merton named it the Matthew effect in 1968 — credit flows to whoever already has some.",
    "PageRank is network centrality: where an endlessly clicking web surfer ends up.",
    "PageRank's imaginary surfer gets bored about 15% of the time and jumps to a page at random.",
    "Stanford owned the PageRank patent, not Google. Google licensed it and paid in shares.",
    "The ARPANET opened in 1969 with four nodes: UCLA, SRI, UC Santa Barbara and Utah.",
    "A follow graph is directed: on Twitter, most friendships only point one way.",
    "Epidemiologists model outbreaks on contact networks, not on crowds of interchangeable people.",
    "Dot and line, vertex and edge, actor and tie, site and bond — four fields, one picture.",
    "This notebook is a network too: change one thing and everything downstream of it redraws itself.",
  ];

  // Lines that belong to ONE module's story — shown only inside it, where the
  // student has met the thing they refer to. The same rule about answers
  // applies here twice over: these land in the module whose checkpoints are
  // about exactly this material.
  const TRIVIA_BY_MODULE: Record<string, string[]> = {
    "m01-euler-tour": [
      "Euler invented network theory in 1736 to settle a stroll — the 7 bridges of Königsberg.",
      "Königsberg's seven bridges went up between 1286 and 1542 — Euler wrote about the set his own century inherited.",
      "Euler filed his bridge paper under a heading of his own invention: 'the geometry of position'.",
      "Königsberg is Kaliningrad now; the city changed its name, and the puzzle kept his.",
      "Euler read his bridge paper to the St Petersburg Academy in 1735; it was printed in 1741.",
    ],
    "m02-small-world": [
      "Frigyes Karinthy dreamed up the six-degrees question in a 1929 short story, 'Chains'.",
      "J. A. Barnes coined 'social network' in 1954, studying a fishing parish in Norway.",
      "Pool and Kochen wrote the small-world problem down around 1958; it sat unpublished for twenty years.",
      "Stanley Milgram ran his letter experiment out of Harvard in the 1960s, with paper and stamps.",
      "Milgram's first packets went out from Wichita, Kansas, before he moved the experiment to Omaha.",
      "Milgram's small-world paper ran in the very first issue of Psychology Today, in 1967.",
      "Most of Milgram's packets never arrived at all — the famous result rests on the ones that did.",
      "In Milgram's Boston study, one clothing merchant handed over a quarter of the letters that arrived.",
      "Milgram's other field experiment: drop stamped letters in the street and see which strangers post them.",
      "In 2003 a team reran Milgram's experiment by email: 24,163 chains, senders in 166 countries.",
      "Watts & Strogatz published their network model in Nature in 1998; it has been cited over 50,000 times.",
      "The Watts-Strogatz paper is three pages long, figures included.",
      "Duncan Watts came to networks from crickets — he was studying how they chirp in sync.",
      "The original three networks studied in 1998: film actors, the US power grid, and a worm's brain.",
      "The C. elegans worm's entire nervous system is mapped — all 302 neurons of it.",
    ],
  };

  const TRIVIA = [...TRIVIA_GENERAL, ...(TRIVIA_BY_MODULE[moduleId()] ?? [])];
  let triviaIdx = Math.floor(Math.random() * TRIVIA.length);
  let triviaTimer: ReturnType<typeof setInterval> | null = null;
  const showTrivia = (ctx: any) => {
    if (quietForPicker) return;
    try {
      const line = `Tip: ${TRIVIA[triviaIdx++ % TRIVIA.length]}`;
      ctx.ui.setWorkingMessage(ctx.ui.theme.fg("dim", line));
    } catch {
      // cosmetic only — never let trivia break a turn
    }
  };
  pi.on("turn_start", async (_event, ctx: any) => {
    if (!ctx.hasUI) return;
    showTrivia(ctx);
    if (triviaTimer) clearInterval(triviaTimer);
    // Long waits (vision calls, compaction) get a new tidbit mid-spin.
    triviaTimer = setInterval(() => showTrivia(ctx), 12_000);
    (triviaTimer as any).unref?.(); // never keep the process alive (print mode)
  });
  pi.on("turn_end", async () => {
    if (triviaTimer) {
      clearInterval(triviaTimer);
      triviaTimer = null;
    }
    // chapter_done arms compaction and leaves the firing to us. It has to be
    // turn_end, not message_end: message_end lands as soon as the tool-calling
    // message is complete, which is BEFORE the tutor says the bridge sentence
    // that same tool result asks for — so compaction aborted it and the
    // student read "Error: This operation was aborted" instead. A turn ends
    // once the tutor has actually stopped talking.
    runPendingCompaction();
    // ── Telling the student the ⚖️ box exists ────────────────────────────
    // The appeal box is the student's only way out from under a tutor that
    // has stopped helping, and nothing ever mentions it to them: AGENTS.md
    // tells the tutor how to OBEY a verdict and never to offer one, the
    // lesson scripts do not name it, and in the notebook it is a collapsed
    // accordion at the bottom of a long page. A beginner going round the
    // same question for the fifth time does not know it is there.
    //
    // "Going round" is countable without asking the model to notice: turns
    // spent inside one checkpoint. A checkpoint normally takes a handful; a
    // dozen means the two of them are stuck, whatever the tutor believes.
    // Said once per checkpoint, as an instruction to mention it in passing —
    // not as an accusation, and not as a reason to stop teaching.
    turnsInCheckpoint += 1;
    if (turnsInCheckpoint === STUCK_TURNS) {
      try {
        pi.sendMessage(
          {
            customType: "stuck-nudge",
            content:
              `NOTE (invisible to the student): you have been on this checkpoint for a ` +
              `while. Keep going exactly as you are — but in your NEXT message, add one ` +
              `plain sentence telling them the ⚖️ box at the bottom of the notebook page ` +
              `is there if they think their answer should count, want a fresh try, or ` +
              `would rather move on, and that using it is never held against them. One ` +
              `sentence, said once, then carry on with the question you are on. Do not ` +
              `apologise and do not suggest they are failing.`,
            display: false,
          },
          { deliverAs: "nextTurn" },
        );
      } catch {
        /* a nudge that cannot be sent is not worth a broken turn */
      }
    }
  });
}
