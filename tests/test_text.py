import logging
from src.crisp_t.text import Text

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_initialization(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    assert text.corpus == corpus_fixture, "Corpus should be set correctly"


def test_common_words(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    common_words = text.common_words(index=5)
    print("Common words:", common_words)
    # Check if the common words are returned as expected
    # [('theory', 4), ('evaluation', 2), ('glaser', 1), ('classical', 1), ('number', 1)]
    assert isinstance(common_words, list), "Common words should be a list"
    assert len(common_words) == 5, "Common words should contain 5 items"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in common_words
    ), "Each item should be a tuple of (word, count)"
    assert all(
        isinstance(item[0], str) and isinstance(item[1], int) for item in common_words
    ), "Each tuple should contain a string and an integer"


def test_common_nouns(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    common_nouns = text.common_nouns(index=5)
    print("Common nouns:", common_nouns)
    # Check if the common nouns are returned as expected
    # [('theory', 4), ('evaluation', 2), ('number', 1), ('guideline', 1), ('methodology', 1)]
    assert isinstance(common_nouns, list), "Common nouns should be a list"
    assert len(common_nouns) == 5, "Common nouns should contain 5 items"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in common_nouns
    ), "Each item should be a tuple of (word, count)"
    assert all(
        isinstance(item[0], str) and isinstance(item[1], int) for item in common_nouns
    ), "Each tuple should contain a string and an integer"


def test_common_verbs(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    common_verbs = text.common_verbs(index=3)
    print("Common verbs:", common_verbs)
    # Check if the common verbs are returned as expected
    # [('theory', 4), ('evaluation', 2), ('number', 1), ('guideline', 1), ('methodology', 1)]
    assert isinstance(common_verbs, list), "Common verbs should be a list"
    assert len(common_verbs) == 3, "Common verbs should contain 3 items"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in common_verbs
    ), "Each item should be a tuple of (word, count)"
    assert all(
        isinstance(item[0], str) and isinstance(item[1], int) for item in common_verbs
    ), "Each tuple should contain a string and an integer"


def test_sentences_with_common_nouns(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    sentences_with_common_nouns = text.sentences_with_common_nouns(index=5)
    print("Sentences with common nouns:", sentences_with_common_nouns)
    # Check if the sentences with common nouns are returned as expected
    assert isinstance(
        sentences_with_common_nouns, list
    ), "Sentences with common nouns should be a list"
    assert (
        len(sentences_with_common_nouns) > 0
    ), "Sentences with common nouns should contain at least one item"
    assert all(
        isinstance(item, str) for item in sentences_with_common_nouns
    ), "Each item should be a string"


def test_spans_with_common_nouns(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    spans_with_common_nouns = text.spans_with_common_nouns(word="evaluation")
    print("Spans with common nouns:", spans_with_common_nouns)
    # Check if the spans with common nouns are returned as expected
    assert isinstance(
        spans_with_common_nouns, list
    ), "Spans with common nouns should be a list"


def test_dimensions(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    dimensions = text.dimensions(word="theory", index=3)
    print("Dimensions:", dimensions)
    # Check if the dimensions are returned as expected
    assert isinstance(dimensions, list), "Dimensions should be a list"
    assert len(dimensions) == 3, "Dimensions should contain 3 items"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in dimensions
    ), "Each item should be a tuple of (word, count)"
    assert all(
        isinstance(item[0], str) and isinstance(item[1], int) for item in dimensions
    ), "Each tuple should contain a string and an integer"


def test_attributes(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    attributes = text.attributes(word="fit", index=3)
    print("Attributes:", attributes)
    # Check if the attributes are returned as expected
    assert isinstance(attributes, list), "Attributes should be a list"
    assert len(attributes) == 3, "Attributes should contain 3 items"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in attributes
    ), "Each item should be a tuple of (word, count)"
    assert all(
        isinstance(item[0], str) and isinstance(item[1], int) for item in attributes
    ), "Each tuple should contain a string and an integer"


def test_generate_summary(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    summary = text.generate_summary(weight=10)
    print("Summary:", summary)
    # Check if the summary is returned as expected
    assert isinstance(summary, list), "Summary should be a list"
    assert len(summary) > 0, "Summary should contain at least one item"
    assert all(
        isinstance(item, str) for item in summary
    ), "Each item should be a string"


def test_print_categories(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    categories = text.print_categories()
    print("Categories:", categories)
    # Check if the categories are returned as expected
    assert isinstance(categories, list), "Categories should be a list"
    assert len(categories) > 0, "Categories should contain at least one item"
    assert all(
        isinstance(item, str) for item in categories
    ), "Each item should be a string"


def test_category_basket(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    category_basket = text.category_basket()
    print("Category basket:", category_basket)
    # Check if the category basket is returned as expected
    assert isinstance(category_basket, list), "Category basket should be a list"
    assert len(category_basket) > 0, "Category basket should contain at least one item"


def test_filter_documents(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    filtered_documents = text.filter_documents(
        metadata_key="file_name", metadata_value="interview-1.txt"
    )
    print("Filtered documents:", filtered_documents)
    # Check if the filtered documents are returned as expected
    assert isinstance(filtered_documents, list), "Filtered documents should be a list"
    assert (
        len(filtered_documents) > 0
    ), "Filtered documents should contain at least one item"
    assert text.initial_document_count > 4
    assert text.document_count() < text.initial_document_count


def test_category_association(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    _category_basket = text.category_basket()
    category_association = text.category_association()
    print("Category association:", category_association)


def test_print_coding_dictionary(corpus_fixture):
    text = Text(corpus=corpus_fixture)
    text.make_spacy_doc()
    coding_dictionary = text.print_coding_dictionary()
    assert "theory" or "coding" in str(coding_dictionary)
