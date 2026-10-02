import { describe, expect, it } from "vitest";

import type { SessionRef } from "@/lib/types";
import {
  candidateParentSessionIds,
  openSubagent,
  openSubagentDetailFromEvent,
} from "@/app/SubagentOpen";

function session(overrides: Partial<SessionRef>): SessionRef {
  return {
    id: "session-1",
    provider: "codex",
    title: "Session",
    project_name: "project",
    is_sidechain: false,
    ...overrides,
  };
}

describe("candidateParentSessionIds", () => {
  it("uses the explicit parent session when present", () => {
    expect(
      candidateParentSessionIds({ parentSessionId: "parent-explicit" }, [
        "active-1",
        "active-2",
      ]),
    ).toEqual(["parent-explicit"]);
  });

  it("falls back to non-empty active parent ids", () => {
    expect(candidateParentSessionIds({}, ["active-1", "", "active-2"])).toEqual(
      ["active-1", "active-2"],
    );
  });
});

describe("openSubagentDetailFromEvent", () => {
  it("extracts only string detail fields", () => {
    const event = new Event("open-subagent") as Event & {
      detail: Record<string, unknown>;
    };
    event.detail = {
      agentId: "agent-1",
      nickname: "Faraday",
      description: "inspect repo",
      parentSessionId: 42,
    };

    expect(openSubagentDetailFromEvent(event)).toEqual({
      agentId: "agent-1",
      nickname: "Faraday",
      description: "inspect repo",
      parentSessionId: undefined,
    });
  });
});

describe("openSubagent", () => {
  it("opens the first child matching the requested agent id", async () => {
    const opened: SessionRef[] = [];
    const loadErrors: string[] = [];
    const parentCalls: string[] = [];
    let activeParentReads = 0;
    let notFound = 0;
    let loadFailed = 0;

    await openSubagent(
      { agentId: "child-2", parentSessionId: "parent-1" },
      {
        getActiveParentSessionIds: () => {
          activeParentReads += 1;
          return ["unused-active"];
        },
        getChildSessions: async (parentId) => {
          parentCalls.push(parentId);
          return [
            session({ id: "agent-child-1", title: "First child" }),
            session({ id: "agent-child-2", title: "Second child" }),
          ];
        },
        getSessionMeta: async (id) => {
          throw new Error(`session not found: ${id}`);
        },
        openSession: (child) => opened.push(child),
        onLoadFailed: () => {
          loadFailed += 1;
        },
        onNotFound: () => {
          notFound += 1;
        },
        onChildSessionLoadError: (parentId) => loadErrors.push(parentId),
      },
    );

    expect(parentCalls).toEqual(["parent-1"]);
    expect(activeParentReads).toBe(0);
    expect(opened.map((child) => child.id)).toEqual(["agent-child-2"]);
    expect(loadErrors).toEqual([]);
    expect(loadFailed).toBe(0);
    expect(notFound).toBe(0);
  });

  it("continues across active parents after a lookup failure", async () => {
    const opened: SessionRef[] = [];
    const loadErrors: string[] = [];

    await openSubagent(
      { description: "full task description" },
      {
        getActiveParentSessionIds: () => ["parent-fails", "parent-works"],
        getChildSessions: async (parentId) => {
          if (parentId === "parent-fails") {
            throw new Error("IPC failed");
          }
          return [
            session({
              id: "child-1",
              title: "full task description",
            }),
          ];
        },
        getSessionMeta: async (id) => {
          throw new Error(`session not found: ${id}`);
        },
        openSession: (child) => opened.push(child),
        onLoadFailed: () => {
          throw new Error("should not report total load failure");
        },
        onNotFound: () => {
          throw new Error("should find a child");
        },
        onChildSessionLoadError: (parentId) => loadErrors.push(parentId),
      },
    );

    expect(loadErrors).toEqual(["parent-fails"]);
    expect(opened.map((child) => child.id)).toEqual(["child-1"]);
  });

  it("reports load failure when every parent lookup errors", async () => {
    let loadFailed = 0;
    let notFound = 0;

    await openSubagent(
      { nickname: "Faraday" },
      {
        getActiveParentSessionIds: () => ["parent-1"],
        getChildSessions: async () => {
          throw new Error("IPC failed");
        },
        getSessionMeta: async (id) => {
          throw new Error(`session not found: ${id}`);
        },
        openSession: () => {
          throw new Error("should not open a child");
        },
        onLoadFailed: () => {
          loadFailed += 1;
        },
        onNotFound: () => {
          notFound += 1;
        },
      },
    );

    expect(loadFailed).toBe(1);
    expect(notFound).toBe(0);
  });

  it("reports not found when parent lookups succeed without a match", async () => {
    let loadFailed = 0;
    let notFound = 0;

    await openSubagent(
      { nickname: "Faraday" },
      {
        getActiveParentSessionIds: () => ["parent-1"],
        getChildSessions: async () => [
          session({ id: "child-1", title: "Ada" }),
        ],
        getSessionMeta: async (id) => {
          throw new Error(`session not found: ${id}`);
        },
        openSession: () => {
          throw new Error("should not open a child");
        },
        onLoadFailed: () => {
          loadFailed += 1;
        },
        onNotFound: () => {
          notFound += 1;
        },
      },
    );

    expect(loadFailed).toBe(0);
    expect(notFound).toBe(1);
  });

  it("opens a child outside the parent's children by its exact session id", async () => {
    // A forked session's inherited delegation belongs to the session it was
    // forked from, so the fork's own children never include it.
    const opened: SessionRef[] = [];
    let notFound = 0;

    await openSubagent(
      { agentId: "child-of-origin", parentSessionId: "fork-1" },
      {
        getActiveParentSessionIds: () => [],
        getChildSessions: async () => [],
        getSessionMeta: async (id) => ({
          ...session({ id, provider: "dsh", is_sidechain: true }),
          parent_id: "origin-1",
          project_path: "/p",
          created_at: 0,
          updated_at: 0,
          message_count: 1,
          file_size_bytes: 0,
          source_path: "/p/s",
          input_tokens: 0,
          output_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
        }),
        openSession: (child) => opened.push(child),
        onLoadFailed: () => {
          throw new Error("should not report load failure");
        },
        onNotFound: () => {
          notFound += 1;
        },
      },
    );

    expect(opened.map((child) => child.id)).toEqual(["child-of-origin"]);
    expect(notFound).toBe(0);
  });
});
