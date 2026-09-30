// Run with Node 18+: node scripts/test_lan_ui_stream.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../pocket_llm_src/app/src/main/assets/lan_ui.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Validate the entire UI script too.
const source = script.slice(script.indexOf('    async function apiResponse('), script.indexOf('    async function loadHistory('));
function client(fetch) {
  const elements = {};
  const context = vm.createContext({ fetch, TextDecoder, sessionToken: 'test',
    sessionStorage: { removeItem() {} }, loginView: {}, chatView: {},
    el: id => elements[id] ||= {} });
  vm.runInContext(source, context);
  return context;
}
const encoder = new TextEncoder();
const event = text => 'data: ' + JSON.stringify({ choices: [{ delta: { content: text } }] }) + '\r\n\r\n';
function streamed(text, byteStep = 1) {
  const bytes = encoder.encode(text);
  return new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += byteStep) controller.enqueue(bytes.slice(i, i + byteStep));
    controller.close();
  } }), { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } });
}
async function main() {
  let controller;
  let firstToken;
  const first = new Promise(resolve => { firstToken = resolve; });
  const updates = [];
  const context = client(async (url, options) => {
    assert.equal(url, '/v1/chat/completions');
    assert.equal(JSON.parse(options.body).stream, true);
    assert.equal(options.headers.Authorization, 'Bearer test');
    return new Response(new ReadableStream({ start(value) { controller = value; } }),
      { headers: { 'Content-Type': 'text/event-stream' } });
  });
  let finished = false;
  const result = context.streamCompletion({ messages: [] }, text => { updates.push(text); firstToken(); })
    .then(answer => { finished = true; return answer; });
  await Promise.resolve();
  controller.enqueue(encoder.encode(event('Hello')));
  await first;
  assert.equal(finished, false, 'Text must appear before the response completes');
  controller.enqueue(encoder.encode(event(' 🌍') + 'data: [DONE]\n\n'));
  assert.equal(await result, 'Hello 🌍');
  assert.deepEqual(updates, ['Hello', 'Hello 🌍']);

  for (const step of [1, 2, 7, 4096]) {
    const c = client(async () => streamed(': heartbeat\n\n' + event('你好 🌍') + event('\nNext') + 'data: [DONE]\n\n', step));
    assert.equal(await c.streamCompletion({}, () => {}), '你好 🌍\nNext');
  }
  const partial = [];
  await assert.rejects(client(async () => streamed(event('Partial'))).streamCompletion({}, text => partial.push(text)), /Connection lost/);
  assert.deepEqual(partial, ['Partial']);
  await assert.rejects(client(async () => streamed(event('Partial') + 'data: {"error":{"message":"Model busy"}}\n\ndata: [DONE]\n\n'))
    .streamCompletion({}, () => {}), /Model busy/);
  await assert.rejects(client(async () => streamed('data: [DONE]\n\n')).streamCompletion({}, () => {}), /no assistant message/);
  await assert.rejects(client(async () => new Response('{"error":{"message":"Unavailable"}}', { status: 503 }))
    .streamCompletion({}, () => {}), /Unavailable/);
  const expired = client(async () => new Response('{}', { status: 401 }));
  await assert.rejects(expired.streamCompletion({}, () => {}), /Session expired/);
  assert.equal(expired.sessionToken, null);
  assert.equal(expired.loginView.hidden, false);
  await assert.rejects(client(async () => new Response('{}', { headers: { 'Content-Type': 'application/json' } }))
    .streamCompletion({}, () => {}), /did not return a response stream/);
  console.log('PASS: incremental delivery, fragmented UTF-8/SSE, heartbeats, EOF/error handling, authentication, and UI syntax');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
