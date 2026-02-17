"""
Tests for TrueWordCloud
"""

import pytest
from PIL import Image
from truewordcloud import TrueWordCloud, WordBox


class TestWordBox:
    """Tests for WordBox dataclass"""

    def test_wordbox_creation(self):
        """Test creating a WordBox"""
        wb = WordBox(
            word="test", frequency=100, font_size=50, width=100, height=30, x=200, y=150
        )
        assert wb.word == "test"
        assert wb.frequency == 100
        assert wb.font_size == 50

    def test_wordbox_bbox(self):
        """Test bounding box calculation"""
        wb = WordBox(
            word="test", frequency=100, font_size=50, width=100, height=30, x=200, y=150
        )
        left, top, right, bottom = wb.bbox
        assert left == 150.0  # 200 - 100/2
        assert top == 135.0  # 150 - 30/2
        assert right == 250.0  # 200 + 100/2
        assert bottom == 165.0  # 150 + 30/2

    def test_wordbox_overlaps(self):
        """Test overlap detection using TrueWordCloud._boxes_overlap"""
        wb1 = WordBox(
            word="test1",
            frequency=100,
            font_size=50,
            width=100,
            height=30,
            x=200,
            y=150,
        )
        wb2 = WordBox(
            word="test2", frequency=80, font_size=40, width=80, height=25, x=250, y=150
        )
        twc = TrueWordCloud(values={})
        # These boxes overlap
        assert twc._boxes_overlap(wb1, wb2)

        # Non-overlapping boxes
        wb3 = WordBox(
            word="test3", frequency=60, font_size=30, width=60, height=20, x=400, y=150
        )
        assert not twc._boxes_overlap(wb1, wb3)


class TestTrueWordCloud:
    """Tests for TrueWordCloud class"""

    def test_initialization_default(self):
        """Test initialization with default parameters"""
        values = {"word1": 100, "word2": 50}
        twc = TrueWordCloud(values=values)
        assert twc.values == values
        assert twc.method == "distance_transform"
        assert twc.base_font_size == 100
        assert twc.min_font_size == 10
        assert twc.background_color == (255, 255, 255)
        assert twc.margin == 2

    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        values = {"word1": 100, "word2": 50}
        twc = TrueWordCloud(
            values=values,
            method="square",
            base_font_size=80,
            min_font_size=15,
            background_color=(0, 0, 0),
            margin=5,
        )
        assert twc.method == "square"
        assert twc.base_font_size == 80
        assert twc.min_font_size == 15
        assert twc.background_color == (0, 0, 0)
        assert twc.margin == 5

    def test_invalid_method(self):
        """Test that invalid method raises error"""
        values = {"word1": 100, "word2": 50}
        with pytest.raises(
            ValueError,
            match="method must be 'greedy', 'square', or 'distance_transform'",
        ):
            TrueWordCloud(values=values, method="invalid")

    def test_calculate_font_sizes(self):
        """Test true proportional font size calculation"""
        values = {"word1": 100, "word2": 50, "word3": 25}
        twc = TrueWordCloud(values=values, base_font_size=100)
        font_sizes = twc._calculate_font_sizes()

        # Check proportionality
        assert font_sizes["word1"] == 100  # max value
        assert font_sizes["word2"] == 50  # half of max
        assert font_sizes["word3"] == 25  # quarter of max

    def test_calculate_font_sizes_with_min(self):
        """Test font size calculation respects minimum"""
        values = {"word1": 100, "word2": 2}
        twc = TrueWordCloud(values=values, base_font_size=100, min_font_size=10)
        font_sizes = twc._calculate_font_sizes()

        assert font_sizes["word1"] == 100
        assert font_sizes["word2"] == 10  # Should be clamped to min_font_size

    def test_generate_greedy(self):
        """Test image generation with greedy method"""
        values = {"python": 100, "data": 80, "science": 60}
        twc = TrueWordCloud(values=values, method="greedy")
        image = twc.generate()

        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0

    def test_generate_square(self):
        """Test image generation with square method"""
        values = {"python": 100, "data": 80, "science": 60}
        twc = TrueWordCloud(values=values, method="square")
        image = twc.generate()

        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0

    def test_generate_with_stats(self):
        """Test generation with statistics"""
        values = {"python": 100, "data": 50, "science": 25}
        twc = TrueWordCloud(values=values, method="greedy")
        image, stats = twc.generate_with_stats()

        assert isinstance(image, Image.Image)
        assert isinstance(stats, dict)
        assert stats["num_words"] == 3
        assert stats["method"] == "greedy"
        assert "size_range" in stats
        assert "canvas_size" in stats
        assert stats["size_range"][0] == 25  # min font size
        assert stats["size_range"][1] == 100  # max font size

    def test_custom_color_func(self):
        """Test custom color function"""

        def red_color(word, freq, norm_freq):
            return (255, 0, 0)

        values = {"word": 100}
        twc = TrueWordCloud(values=values, color_func=red_color)
        # Should not raise an error
        image = twc.generate()
        assert isinstance(image, Image.Image)

    def test_empty_values(self):
        """Test handling of empty values dict"""
        twc = TrueWordCloud(values={})
        font_sizes = twc._calculate_font_sizes()
        assert font_sizes == {}

    def test_single_word(self):
        """Test with a single word"""
        values = {"solo": 100}
        twc = TrueWordCloud(values=values)
        image = twc.generate()
        assert isinstance(image, Image.Image)

    def test_many_words_greedy(self):
        """Test with many words using greedy method"""
        values = {f"word{i}": 100 - i * 5 for i in range(20)}
        twc = TrueWordCloud(values=values, method="greedy")
        image = twc.generate()
        assert isinstance(image, Image.Image)

    def test_many_words_square(self):
        """Test with many words using square method"""
        values = {f"word{i}": 100 - i * 5 for i in range(20)}
        twc = TrueWordCloud(values=values, method="square")
        image = twc.generate()
        assert isinstance(image, Image.Image)

    def test_proportionality_maintained(self):
        """Test that proportional relationships are maintained"""
        values = {"big": 200, "medium": 100, "small": 50}
        twc = TrueWordCloud(values=values, base_font_size=100)
        font_sizes = twc._calculate_font_sizes()

        # Verify exact 2:1 and 4:1 ratios
        assert font_sizes["big"] / font_sizes["medium"] == 2.0
        assert font_sizes["big"] / font_sizes["small"] == 4.0
        assert font_sizes["medium"] / font_sizes["small"] == 2.0


class TestIntegration:
    """Integration tests"""

    def test_full_workflow_greedy(self):
        """Test complete workflow with greedy method"""
        values = {
            "python": 1000,
            "data": 800,
            "science": 600,
            "machine": 400,
            "learning": 400,
        }

        twc = TrueWordCloud(
            values=values, method="greedy", base_font_size=80, min_font_size=12
        )

        image, stats = twc.generate_with_stats()

        # Verify image
        assert isinstance(image, Image.Image)
        assert image.size[0] > 0
        assert image.size[1] > 0

        # Verify stats
        assert stats["num_words"] == 5
        assert stats["method"] == "greedy"
        assert stats["size_range"][1] == 80  # base_font_size for max value

    def test_full_workflow_square(self):
        """Test complete workflow with square method"""
        values = {
            "alpha": 500,
            "beta": 400,
            "gamma": 300,
            "delta": 200,
        }

        twc = TrueWordCloud(
            values=values,
            method="square",
            base_font_size=60,
            background_color=(240, 240, 240),
        )

        image, stats = twc.generate_with_stats()

        # Verify image
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

        # Verify stats
        assert stats["num_words"] == 4
        assert stats["method"] == "square"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
