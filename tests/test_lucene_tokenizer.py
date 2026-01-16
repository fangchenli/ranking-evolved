"""
Tests for the pure Python Lucene tokenizer implementation.

Compares output with Pyserini's Lucene analyzer (when available) and tests
the Porter stemmer implementation against known outputs.
"""

import pytest

from ranking_evolved.bm25 import (
    ENGLISH_STOPWORDS,
    LuceneTokenizer,
    PorterStemmer,
    lucene_tokenize,
)


class TestPorterStemmer:
    """Test the Porter stemmer implementation against known outputs."""

    @pytest.fixture
    def stemmer(self):
        return PorterStemmer()

    # Test cases from the original Porter algorithm paper
    @pytest.mark.parametrize(
        "word,expected",
        [
            # Step 1a: plurals
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            # Step 1b: -ed, -ing
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("bled", "bled"),
            ("motoring", "motor"),
            ("sing", "sing"),
            # Step 1c: y -> i
            ("happy", "happi"),
            ("sky", "sky"),
            # Step 2: double suffixes
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("valenci", "valenc"),
            ("hesitanci", "hesit"),
            ("digitizer", "digit"),
            ("conformabli", "conform"),
            ("radicalli", "radic"),
            ("differentli", "differ"),
            ("vileli", "vile"),
            ("analogousli", "analog"),
            ("vietnamization", "vietnam"),
            ("predication", "predic"),
            ("operator", "oper"),
            ("feudalism", "feudal"),
            ("decisiveness", "decis"),
            ("hopefulness", "hope"),
            ("callousness", "callous"),
            ("formaliti", "formal"),
            ("sensitiviti", "sensit"),
            ("sensibiliti", "sensibl"),
            # Step 3: -icate, -ative, etc.
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electriciti", "electr"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            # Step 4: -al, -ance, -ence, etc.
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("homologou", "homolog"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("angulariti", "angular"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
            # Step 5a: -e
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            # Step 5b: -ll
            ("controll", "control"),
            ("roll", "roll"),
        ],
    )
    def test_porter_stem(self, stemmer, word, expected):
        """Test individual Porter stemmer cases."""
        assert stemmer.stem(word) == expected

    def test_short_words(self, stemmer):
        """Short words should pass through unchanged."""
        assert stemmer.stem("a") == "a"
        assert stemmer.stem("is") == "is"

    def test_already_stemmed(self, stemmer):
        """Already stemmed words should be unchanged."""
        assert stemmer.stem("run") == "run"
        assert stemmer.stem("cat") == "cat"


class TestLuceneTokenizer:
    """Test the Lucene tokenizer pipeline."""

    @pytest.fixture
    def tokenizer(self):
        return LuceneTokenizer()

    @pytest.fixture
    def tokenizer_no_stem(self):
        return LuceneTokenizer(stem=False)

    def test_basic_tokenization(self, tokenizer):
        """Test basic tokenization with default settings."""
        result = tokenizer("The quick brown fox")
        # "the" is a stopword, "quick", "brown", "fox" are stemmed
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_stopword_removal(self, tokenizer):
        """Test that stopwords are removed."""
        result = tokenizer("this is a test of the system")
        # Most words here are stopwords
        assert "this" not in result
        assert "is" not in result
        assert "a" not in result
        assert "the" not in result
        assert "of" not in result
        assert "test" in result
        assert "system" in result

    def test_possessive_removal(self, tokenizer):
        """Test possessive suffix removal."""
        result = tokenizer("John's book is Mary's")
        assert "john" in result
        assert "book" in result
        assert "mari" in result  # stemmed

    def test_stemming(self, tokenizer):
        """Test that stemming is applied."""
        result = tokenizer("running jumps quickly")
        assert "run" in result
        assert "jump" in result
        assert "quickli" in result  # Porter stems "quickly" to "quickli"

    def test_no_stemming(self, tokenizer_no_stem):
        """Test tokenizer without stemming."""
        result = tokenizer_no_stem("running jumps quickly")
        assert "running" in result
        assert "jumps" in result
        assert "quickly" in result

    def test_lowercase(self, tokenizer):
        """Test that text is lowercased."""
        result = tokenizer("Hello WORLD")
        assert all(token.islower() for token in result)

    def test_empty_string(self, tokenizer):
        """Test empty string input."""
        assert tokenizer("") == []

    def test_only_stopwords(self, tokenizer):
        """Test string with only stopwords."""
        assert tokenizer("the a an is are was") == []

    def test_numbers(self, tokenizer):
        """Test that numbers are preserved."""
        result = tokenizer("chapter 42 section 7")
        assert "42" in result
        assert "7" in result
        assert "chapter" in result
        assert "section" in result

    def test_mixed_content(self, tokenizer):
        """Test mixed alphanumeric content."""
        result = tokenizer("Python3 is great for NLP tasks")
        assert "python3" in result
        assert "great" in result
        assert "nlp" in result
        assert "task" in result  # stemmed

    def test_custom_stopwords(self):
        """Test custom stopword set."""
        custom_stops = frozenset(["custom", "stop"])
        tokenizer = LuceneTokenizer(stopwords=custom_stops)
        result = tokenizer("this is a custom stop test")
        assert "custom" not in result
        assert "stop" not in result
        assert "thi" in result  # "this" stems to "thi", not in custom stopwords
        assert "test" in result

    def test_no_stopwords(self):
        """Test with empty stopword set."""
        tokenizer = LuceneTokenizer(stopwords=frozenset())
        result = tokenizer("the quick brown fox")
        assert "the" in result
        assert "quick" in result


class TestLuceneTokenizeFunction:
    """Test the convenience function."""

    def test_lucene_tokenize(self):
        """Test the module-level convenience function."""
        result = lucene_tokenize("The quick brown fox jumps")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result
        assert "jump" in result  # stemmed


class TestStopwords:
    """Test the stopword set."""

    def test_common_stopwords_present(self):
        """Test that common English stopwords are in the set."""
        common = ["a", "an", "the", "is", "are", "was", "were", "be", "been"]
        for word in common:
            assert word in ENGLISH_STOPWORDS

    def test_content_words_absent(self):
        """Test that content words are not in stopwords."""
        content = ["computer", "science", "python", "algorithm", "data"]
        for word in content:
            assert word not in ENGLISH_STOPWORDS


class TestPyseriniCompatibility:
    """Test compatibility with Pyserini's Lucene analyzer.

    These tests require Pyserini and Java to be installed.
    """

    @pytest.fixture
    def pyserini_analyzer(self):
        """Get Pyserini's Lucene analyzer if available."""
        try:
            from pyserini.analysis import Analyzer, get_lucene_analyzer

            return Analyzer(get_lucene_analyzer())
        except ImportError:
            pytest.skip("Pyserini not installed")
        except Exception as e:
            pytest.skip(f"Pyserini not available: {e}")

    @pytest.fixture
    def our_tokenizer(self):
        return LuceneTokenizer()

    @pytest.mark.parametrize(
        "text",
        [
            "The quick brown fox jumps over the lazy dog",
            "Information retrieval systems",
            "Running and jumping are exercises",
            "Python's programming capabilities",
            "BM25 ranking algorithm implementation",
            "Natural language processing tasks",
        ],
    )
    def test_pyserini_compatibility(self, pyserini_analyzer, our_tokenizer, text):
        """Compare our tokenizer output with Pyserini's.

        Note: Small differences may exist due to:
        1. Different Unicode handling
        2. Slightly different Porter stemmer implementations
        3. Different stopword lists

        We check for high overlap rather than exact match.
        """
        pyserini_tokens = set(pyserini_analyzer.analyze(text))
        our_tokens = set(our_tokenizer(text))

        # Calculate Jaccard similarity
        intersection = pyserini_tokens & our_tokens
        union = pyserini_tokens | our_tokens

        if union:
            similarity = len(intersection) / len(union)
            # We expect at least 70% similarity
            assert similarity >= 0.7, (
                f"Low similarity ({similarity:.2%}): Pyserini={pyserini_tokens}, Ours={our_tokens}"
            )
