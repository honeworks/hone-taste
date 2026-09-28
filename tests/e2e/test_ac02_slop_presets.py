"""AC-2: the lyrics preset catches lyric cliches; the email preset catches email cliches."""

import hone_taste as tt


def _found(score: tt.Score) -> set[str]:
    return {h.get("match", h.get("word", "")).lower() for h in score.details["hits"]}


def _value(score: tt.Score) -> float:
    assert score.value is not None
    return score.value


def test_ac2_slop_presets() -> None:
    lyric = "Under neon lights we dance, echoes of a love we lost, and the city lights keep calling me home."
    email = "Hi Sam, I hope this email finds you well. Let's touch base on Friday about the invoice."

    lyrics, general = tt.slop_score(domain="lyrics"), tt.slop_score()  # lyrics first: presets must not leak
    assert {"neon lights", "echoes of", "city lights"} <= _found(lyrics(lyric))
    assert not {"neon lights", "echoes of", "city lights"} & _found(general(lyric))
    assert _value(lyrics(lyric)) < _value(general(lyric))
    words_line = "Embers fall, a symphony of neon"
    assert {"embers", "symphony", "neon"} <= _found(lyrics(words_line))
    assert not {"embers", "symphony", "neon"} & _found(general(words_line))

    mail = tt.slop_score(domain="email")
    assert {"i hope this email finds you well", "touch base"} <= _found(mail(email))
    assert not {"i hope this email finds you well", "touch base"} & _found(general(email))
    assert _value(mail(email)) < _value(general(email))
    assert mail.name == "slop_email"
