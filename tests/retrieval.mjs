import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {buildIndex,answer,evaluate} from '../dist/engine.mjs';
const docs=JSON.parse(readFileSync(new URL('../dist/corpus.json',import.meta.url)));const index=buildIndex(docs);const rows=evaluate(index);
for(const row of rows)console.log(`${row.pass?'PASS':'FAIL'} ${row.q} => ${row.result.status} ${row.result.hits.map(h=>h.id).join(',')}`);
assert.equal(index.docs.length,13);assert(!index.docs.some(d=>d.archived));assert.equal(rows.filter(r=>r.pass).length,rows.length);assert(rows.every(r=>r.citationIntegrity));assert(!answer(index,'How can I get a refund on an annual subscription?').sources.some(s=>s.archived));console.log(`${rows.length} evaluation cases passed; archived-policy filtering and citation integrity passed.`);
