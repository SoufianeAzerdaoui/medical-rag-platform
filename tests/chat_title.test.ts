import test from "node:test";
import assert from "node:assert/strict";

import { deriveAutoConversationTitle, resolveAutoTitleUpdate } from "../src/lib/chat-title";

test("small talk 'bonjour' ne renomme pas", () => {
  assert.equal(deriveAutoConversationTitle("bonjour", "general"), null);
});

test("anomalies -> Résultats hors référence", () => {
  assert.equal(deriveAutoConversationTitle("Quels résultats semblent anormaux ?", "general"), "Résultats hors référence");
});

test("comparaison générique -> Comparaison de rapports", () => {
  assert.equal(deriveAutoConversationTitle("Compare ces deux documents", "general"), "Comparaison de rapports");
});

test("titre manuel non écrasé", () => {
  const resolved = resolveAutoTitleUpdate({
    currentTitle: "Titre utilisateur",
    titleSource: "manual",
    message: "Quels résultats semblent anormaux ?",
    mode: "general",
    messageCount: 0,
  });
  assert.equal(resolved.title, "Titre utilisateur");
  assert.equal(resolved.titleSource, "manual");
});
