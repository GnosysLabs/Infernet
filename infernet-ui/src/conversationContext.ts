import type { ChatMessage } from "./chatHistory";

// The 32,768-token runtime reserves 4,096 tokens for generation, leaving 28,672
// prompt tokens. This conservative character heuristic retains roughly 20K
// tokens of typical English prose until exact tokenizer budgeting is available.
const MAX_RENDERED_PROMPT_CHARACTERS = 80_000;
const CONTEXT_INSTRUCTION =
  "Continue the conversation below. Use the earlier turns as context and respond to the final User message.";

export function buildConversationPrompt(
  previousMessages: readonly ChatMessage[],
  userPrompt: string,
): string {
  const currentPrompt = userPrompt.trim();
  if (previousMessages.length === 0) return currentPrompt;

  const currentTurn = formatTurn("user", currentPrompt);
  const responseCue = "Assistant:";
  const fixedLength = CONTEXT_INSTRUCTION.length + currentTurn.length + responseCue.length + 4;
  let remainingCharacters = Math.max(0, MAX_RENDERED_PROMPT_CHARACTERS - fixedLength);
  const includedGroups: string[] = [];
  let group: string[] = [];

  for (let index = previousMessages.length - 1; index >= 0; index -= 1) {
    const message = previousMessages[index];
    group.unshift(formatTurn(message.role, message.text));
    if (message.role !== "user" && index !== 0) continue;

    const renderedGroup = group.join("\n\n");
    const renderedLength = renderedGroup.length + 2;
    if (renderedLength > remainingCharacters) break;
    includedGroups.unshift(renderedGroup);
    remainingCharacters -= renderedLength;
    group = [];
  }

  return [
    CONTEXT_INSTRUCTION,
    ...includedGroups,
    currentTurn,
    responseCue,
  ].join("\n\n");
}

function formatTurn(role: ChatMessage["role"], text: string): string {
  const label = role === "assistant" ? "Assistant" : "User";
  return `${label}:\n${text.trim()}`;
}
