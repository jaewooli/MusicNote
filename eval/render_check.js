// Format every measure of a ScoreDoc with VexFlow, headless.
//
// This is the check that the notation layer's arithmetic survives the renderer:
// VexFlow rejects a voice whose ticks do not add up, and it is the only thing
// that knows whether three voices fit on one staff. Called by test_render.py.
//
//   node eval/render_check.js <doc.json> [<doc.json> ...]
const fs = require('fs');
const path = require('path');

global.window = global.window || {};
global.document = {
  createElement: () => ({
    getContext: () => null, style: {}, appendChild() {}, setAttribute() {},
  }),
};
const VF = require(path.join(__dirname, '..', 'frontend', 'vendor', 'vexflow.js'));

const VF_DUR = { breve: '1/2', whole: 'w', half: 'h', quarter: 'q',
  eighth: '8', '16th': '16', '32nd': '32', '64th': '64' };
const VF_HEAD = { x: 'x2', 'circle-x': 'cx', diamond: 'd2' };

function key(n) {
  if (n.unpitched) {
    const head = VF_HEAD[n.notehead];
    return n.step.toLowerCase() + '/' + n.octave + (head ? '/' + head : '');
  }
  const acc = n.alter > 0 ? '#'.repeat(n.alter) : n.alter < 0 ? 'b'.repeat(-n.alter) : '';
  return n.step.toLowerCase() + acc + '/' + n.octave;
}

let bad = 0, bars = 0, staves = 0, maxVoices = 0;
for (const file of process.argv.slice(2)) {
  const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
  const [num, den] = doc.time_sig;
  const nMeas = Math.max(...doc.parts.map(
    p => Math.max(...p.voices.map(v => v.measures.length))));
  for (const part of doc.parts) {
    for (let si = 1; si <= (part.staves || 1); si++) {
      const clef = (part.clefs || ['treble'])[si - 1] || 'treble';
      const src = part.voices.filter(v => (v.staff || 1) === si);
      if (!src.length) continue;
      staves++;
      for (let mi = 0; mi < nMeas; mi++) {
        const show = src.filter(v => v.measures[mi] && v.measures[mi].events.length);
        if (!show.length) continue;
        maxVoices = Math.max(maxVoices, show.length);
        bars++;
        const voices = [];
        for (const v of show) {
          const tickables = v.measures[mi].events.map(e => {
            const code = VF_DUR[e.type] || 'q';
            if (!e.notes || !e.notes.length) {
              return new VF.StaveNote({ clef, keys: [clef === 'bass' ? 'd/3' : 'b/4'],
                duration: code + 'r', dots: e.dots });
            }
            const sn = new VF.StaveNote({ clef, keys: e.notes.map(key), duration: code });
            for (let d = 0; d < (e.dots || 0); d++) VF.Dot.buildAndAttach([sn]);
            return sn;
          });
          // SOFT is the mode the app renders in — a bar that comes up short
          // is drawn, not rejected — so the check has to use it too, or it
          // would be testing a renderer nobody runs.
          const voice = new VF.Voice({ num_beats: num, beat_value: den })
            .setMode(VF.Voice.Mode.SOFT);
          voice.addTickables(tickables);
          voices.push(voice);
        }
        try {
          VF.Accidental.applyAccidentals(voices, 'C');
          new VF.Formatter().joinVoices(voices).format(voices, 400);
        } catch (e) {
          bad++;
          if (bad <= 5) {
            console.log(`FAIL ${path.basename(file)} ${part.id} staff${si} bar${mi + 1} `
              + `voices=${show.length}: ${e.message}`);
          }
        }
      }
    }
  }
}
console.log(`formatted ${bars} bar(s) over ${staves} staff/staves; `
  + `up to ${maxVoices} voice(s) on a staff; ${bad} failure(s)`);
process.exit(bad ? 1 : 0);
