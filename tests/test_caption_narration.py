import json
import re
from pathlib import Path
import pytest
from chord.image_caption import CaptionImages

SAMPLES=json.loads((Path(__file__).parent/'fixtures/voice_caption_artifacts.json').read_text())

@pytest.mark.parametrize('row', SAMPLES, ids=lambda r:f"{r['arm']}-{r['i']}")
def test_saved_artifact_narration_removed_at_every_split(row):
    text=row['text']
    if row['i']==0 and row['arm']=='old':
        expected=text[:text.index('{')]
    else:
        expected=re.sub(r'\[(?:Image|Picture)(?::| of)[^]]*\]', '', text)
    assert expected != text
    for split in range(len(text)+1):
        f=CaptionImages()
        assert f.feed(text[:split])+f.feed(text[split:])+f.feed('',final=True)==expected
    f=CaptionImages()
    assert ''.join(f.feed(c) for c in text)+f.feed('',final=True)==expected

@pytest.mark.parametrize('text', ['{"description":"[Image: a cup]"}', '[note: keep this]', '[Image: incomplete', '[Image: a cup](https://example.com)', '[Picture of a cup][reference]', '[Image quality matters]', 'ordinary [brackets]'])
def test_other_brackets_and_links_preserved(text):
    f=CaptionImages()
    assert ''.join(f.feed(c) for c in text)+f.feed('',final=True)==text
