"""
ScoreDoc → MusicXML 3.1 (partwise).

This is now a *serializer only*. It used to re-derive notation from a flat note
list with its own 16th-note grid and top-note-only reduction — which is why the
export could not express chords, triplets or real durations, and why it drifted
from what the screen showed. All of that now lives in score_build.build_score();
both the screen and this file render the same ScoreDoc.

`build()` keeps the old (notes, tempo, time_sig, title) signature so callers do
not change: it just builds a ScoreDoc first.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from score_model import DIVISIONS, ScoreDoc

_DYN_MARKS = [(24, "pp"), (44, "p"), (64, "mp"), (88, "mf"), (108, "f"), (127, "ff")]


def _dyn_mark(vel: int) -> str:
    for lim, m in _DYN_MARKS:
        if vel <= lim:
            return m
    return "ff"


def _note_xml(n, dur: int, ty: str, dots: int, tup, tup_start: bool,
              tup_stop: bool, chord: bool, voice: int = 1, staff: str = "") -> str:
    """One <note>. MusicXML fixes the child order (… tie, voice, type, dot,
    time-modification, staff, notations), and importers that reject a file
    silently are the reason to follow it exactly."""
    bits = ["<note>"]
    if chord:
        bits.append("<chord/>")
    if n.unpitched:
        # A kit piece, not a pitch. <unpitched> names the line to draw it on and
        # nothing else, so no <alter> and no accidental from the key signature.
        bits.append(f"<unpitched><display-step>{n.step}</display-step>"
                    f"<display-octave>{n.octave}</display-octave></unpitched>")
    else:
        alter = f"<alter>{n.alter}</alter>" if n.alter else ""
        bits.append(f"<pitch><step>{n.step}</step>{alter}"
                    f"<octave>{n.octave}</octave></pitch>")
    bits.append(f"<duration>{dur}</duration>")
    if n.tie_stop:
        bits.append('<tie type="stop"/>')
    if n.tie_start:
        bits.append('<tie type="start"/>')
    bits.append(f"<voice>{voice}</voice>")
    bits.append(f"<type>{ty}</type>")
    bits.append("<dot/>" * dots)
    if n.notehead and n.notehead != "normal":
        bits.append(f"<notehead>{n.notehead}</notehead>")
    if tup:
        bits.append(f"<time-modification><actual-notes>{tup[0]}</actual-notes>"
                    f"<normal-notes>{tup[1]}</normal-notes></time-modification>")
    bits.append(staff)
    nots = []
    if n.tie_stop:
        nots.append('<tied type="stop"/>')
    if n.tie_start:
        nots.append('<tied type="start"/>')
    if tup and tup_start:
        nots.append('<tuplet type="start" bracket="yes"/>')
    if tup and tup_stop:
        nots.append('<tuplet type="stop"/>')
    if nots:
        bits.append("<notations>" + "".join(nots) + "</notations>")
    bits.append("</note>")
    return "".join(bits)


def doc_to_musicxml(doc: ScoreDoc) -> str:
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 '
        'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        f'<work><work-title>{escape(str(doc.title))}</work-title></work>',
        '<identification><encoding><software>MusicNote</software>'
        '</encoding></identification>',
        '<part-list>',
    ]
    for p in doc.parts:
        out.append(f'<score-part id="{p.id}"><part-name>{escape(p.name)}</part-name>'
                   f'<score-instrument id="{p.id}-I"><instrument-name>'
                   f'{escape(p.name)}</instrument-name></score-instrument>'
                   f'<midi-instrument id="{p.id}-I">'
                   + ("<midi-channel>10</midi-channel>" if p.is_drum else "")
                   + f'<midi-program>{int(p.program) + 1}</midi-program>'
                   f'</midi-instrument></score-part>')
    out.append('</part-list>')

    for p in doc.parts:
        out.append(f'<part id="{p.id}">')
        n_meas = max((len(v.measures) for v in p.voices), default=0)
        for mi in range(n_meas):
            out.append(f'<measure number="{mi + 1}">')
            if mi == 0:
                clefs = list(p.clefs or [p.clef])[:max(1, p.staves)]
                while len(clefs) < max(1, p.staves):
                    clefs.append(p.clef)
                signs = {"bass": "<sign>F</sign><line>4</line>",
                         "percussion": "<sign>percussion</sign><line>2</line>",
                         "treble": "<sign>G</sign><line>2</line>"}
                cx = "".join(
                    f'<clef number="{i + 1}">'
                    + signs.get(c, signs["treble"]) + "</clef>"
                    for i, c in enumerate(clefs))
                staves = (f'<staves>{p.staves}</staves>' if p.staves > 1 else "")
                # A percussion staff carries no key: its lines are positions,
                # not pitches, so a key signature there is meaningless.
                key = ("" if p.is_drum
                       else f'<key><fifths>{doc.key_fifths}</fifths></key>')
                out.append(
                    f'<attributes><divisions>{doc.divisions}</divisions>'
                    f'{key}'
                    f'<time><beats>{doc.time_sig[0]}</beats>'
                    f'<beat-type>{doc.time_sig[1]}</beat-type></time>'
                    f'{staves}{cx}</attributes>')
                bpm = int(round(doc.tempo))
                out.append('<direction placement="above"><direction-type><metronome>'
                           '<beat-unit>quarter</beat-unit>'
                           f'<per-minute>{bpm}</per-minute></metronome>'
                           f'</direction-type><sound tempo="{bpm}"/></direction>')

            last_dyn = 100
            for vi, v in enumerate(p.voices):
                if vi:                       # rewind for the next voice
                    total = sum(c.dur for c in v.measures[mi].events) \
                        if mi < len(v.measures) else 0
                    prev = sum(c.dur for c in p.voices[vi - 1].measures[mi].events) \
                        if mi < len(p.voices[vi - 1].measures) else 0
                    if prev:
                        out.append(f'<backup><duration>{prev}</duration></backup>')
                    if not total:
                        continue
                if mi >= len(v.measures):
                    continue
                staff_x = f"<staff>{v.staff}</staff>" if p.staves > 1 else ""
                for c in v.measures[mi].events:
                    if c.is_rest:
                        out.append(f'<note><rest/><duration>{c.dur}</duration>'
                                   f'<voice>{v.number}</voice><type>{c.type}</type>'
                                   + "<dot/>" * c.dots + staff_x + '</note>')
                        continue
                    vel = max(n.velocity for n in c.notes)
                    dyn = max(5, min(200, round(vel / 90 * 100)))
                    if abs(dyn - last_dyn) >= 12 and vi == 0:
                        out.append('<direction placement="below"><direction-type>'
                                   f'<dynamics><{_dyn_mark(vel)}/></dynamics>'
                                   f'</direction-type><sound dynamics="{dyn}"/>'
                                   '</direction>')
                        last_dyn = dyn
                    for k, n in enumerate(c.notes):
                        out.append(_note_xml(
                            n, c.dur, c.type, c.dots, c.tuplet, c.tuplet_start,
                            c.tuplet_stop, chord=k > 0, voice=v.number,
                            staff=staff_x))
            out.append('</measure>')
        out.append('</part>')
    out.append('</score-partwise>')
    return "\n".join(out)


def build(notes, tempo: float = 120.0, time_sig=(4, 4),
          title: str = "MusicNote", beats=None, parts=None) -> str:
    """Backwards-compatible entry point: notes → ScoreDoc → MusicXML.
    `parts` (list of {name, notes, voices, program}) takes precedence over the
    flat `notes` list, so a multi-instrument score exports every part."""
    from score_build import build_score
    pin = parts or [{"name": "Music", "notes": list(notes or [])}]
    doc = build_score(pin, beats=beats, tempo=tempo, time_sig=time_sig, title=title)
    return doc_to_musicxml(doc)
