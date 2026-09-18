import rag

URL = "https://www.legislation.gov.uk/ukpga/1968/60/data.xml"

# A cut-down Act, written to match the real XML structure and carry every
# awkward case the loader has to handle:
#   - section 1(1), plain subsection
#   - section 9(1), nested (a)/(b) points whose letters must not glue to the text
#   - section 9(2), a subsection with no DocumentURI: text quoted from another
#     Act, which must be skipped rather than indexed as this Act's law
#   - Pnumber holding CommentaryRef tags before the digit, so .text is None
#   - a Schedule paragraph, which must be labelled Paragraph, not Section
SAMPLE_XML = """<Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation">
  <Primary><Body>
    <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/section/1" id="section-1">
      <Pnumber>1</Pnumber>
      <P1para>
        <P2 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/section/1/1" id="section-1-1">
          <Pnumber>1</Pnumber>
          <P2para><Text>A person is guilty of theft if he dishonestly appropriates property belonging to another with the intention of permanently depriving the other of it.</Text></P2para>
        </P2>
      </P1para>
    </P1>

    <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/section/9" id="section-9">
      <Pnumber><CommentaryRef Ref="c1"/><CommentaryRef Ref="c2"/>9</Pnumber>
      <P1para>
        <P2 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/section/9/1" id="section-9-1">
          <Pnumber>1</Pnumber>
          <P2para>
            <Text>A person is guilty of burglary if&#8212;</Text>
            <P3 id="section-9-1-a"><Pnumber>a</Pnumber><P3para><Text>he enters any building as a trespasser with intent to commit an offence; or</Text></P3para></P3>
            <P3 id="section-9-1-b"><Pnumber>b</Pnumber><P3para><Text>having entered as a trespasser he steals anything in the building.</Text></P3para></P3>
          </P2para>
        </P2>
        <P2 id="quoted-from-another-act">
          <Pnumber>1</Pnumber>
          <P2para><Text>Quoted text inserted into a different Act, which this loader must not index.</Text></P2para>
        </P2>
      </P1para>
    </P1>

    <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/schedule/1/paragraph/1" id="schedule-1-paragraph-1">
      <Pnumber>1</Pnumber>
      <P1para>
        <P2 DocumentURI="http://www.legislation.gov.uk/ukpga/1968/60/schedule/1/paragraph/1/1" id="schedule-1-paragraph-1-1">
          <Pnumber>1</Pnumber>
          <P2para><Text>Offences of taking game or fish are triable summarily.</Text></P2para>
        </P2>
      </P1para>
    </P1>
  </Body></Primary>
</Legislation>"""


def parse():
    # Every test starts from the same sample, so none of them can affect another
    from xml.etree import ElementTree as ET
    return rag.split(ET.fromstring(SAMPLE_XML), URL)


def find(docs, section, subsection):
    # Returns the one chunk matching a section and subsection, or None
    for d in docs:
        if d.metadata["section"] == section and d.metadata["subsection"] == subsection:
            return d
    return None


def test_chunks_are_well_formed():
    # Shape check: every chunk has an ID, a unique ID, and text behind its citation
    docs = parse()

    assert docs, "no chunks produced"

    for d in docs:
        assert d.metadata["uri"], f"missing uri: {d.page_content[:40]}"
        text = d.page_content.split(": ", 1)[1]
        assert text.strip(), f"no text: {d.page_content[:40]}"

    uris = set()
    for d in docs:
        uris.add(d.metadata["uri"])
    assert len(uris) == len(docs), "duplicate URIs"


def test_plain_section_parses():
    # Normal case: one subsection, no nesting
    docs = parse()
    found = find(docs, "1", "1")

    assert found is not None, "section 1(1) not produced"
    assert found.page_content.startswith("Section 1(1):")
    assert "dishonestly appropriates" in found.page_content


def test_section_number_survives_nested_commentary():
    # Pnumber holds CommentaryRef tags before the digit, so .text is None
    # and only itertext() returns the number
    docs = parse()

    sections = set()
    for d in docs:
        sections.add(d.metadata["section"])

    assert "9" in sections, "section number lost to nested tags"
    assert "None" not in sections


def test_nested_points_are_included():
    # Subsection with lettered sub-points: the text must be pulled in but the
    # letters themselves must not glue onto the sentence
    docs = parse()
    found = find(docs, "9", "1")

    assert found is not None, "section 9(1) not produced"
    assert "trespasser" in found.page_content
    assert "ahe enters" not in found.page_content, "sub-point letter glued to text"


def test_quoted_text_from_other_acts_is_skipped():
    # A P2 with no DocumentURI is text quoted into a different Act. Indexing it
    # would attribute another Act's law to this one.
    docs = parse()

    for d in docs:
        assert "must not index" not in d.page_content, "quoted text was indexed"


def test_schedule_paragraphs_are_not_called_sections():
    # Schedules number their provisions as paragraphs. Labelling them "Section"
    # produced two different provisions sharing one citation.
    docs = parse()

    schedule = []
    for d in docs:
        if "/schedule/" in d.metadata["uri"]:
            schedule.append(d)

    assert schedule, "no schedule provisions in sample"

    for d in schedule:
        assert d.page_content.startswith("Paragraph"), f"mislabelled: {d.page_content[:40]}"
        assert d.metadata["type"] == "paragraph"