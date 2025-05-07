from gridworld.components.grid_environment import VisitCounter


class TestGetItem:
    def test_get_item(self):
        visit_counter = VisitCounter()
        visit_counter[(0, 0)] = 1
        assert visit_counter[(0, 0)] == 1
        assert visit_counter[(1, 1)] == 0


class TestSetItem:
    def test_set_item(self):
        visit_counter = VisitCounter()
        visit_counter[(0, 0)] = 1
        assert visit_counter[(0, 0)] == 1
        visit_counter[(0, 0)] = 2
        assert visit_counter[(0, 0)] == 2
        visit_counter[(1, 1)] = 3
        assert visit_counter[(1, 1)] == 3

    def test_increment_item(self):
        visit_counter = VisitCounter()
        visit_counter[(0, 0)] += 1
        assert visit_counter[(0, 0)] == 1
        visit_counter[(0, 0)] += 1
        assert visit_counter[(0, 0)] == 2


class TestEquals:
    def test_equals(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 1
        visit_counter2[(1, 1)] = 2

        assert visit_counter1 == visit_counter2

    def test_not_equals(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 1
        visit_counter2[(1, 1)] = 3

        assert visit_counter1 != visit_counter2

    def test_compares_to_dict(self):
        visit_counter = VisitCounter()
        visit_counter[(0, 0)] = 1
        visit_counter[(1, 1)] = 2

        assert visit_counter == {(0, 0): 1, (1, 1): 2}
        assert visit_counter != {(0, 0): 1, (1, 1): 3}


class TestAdd:
    def test_add(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 3
        visit_counter2[(1, 1)] = 4

        result = visit_counter1 + visit_counter2

        assert result == {
            (0, 0): 4,
            (1, 1): 6,
        }

    def test_handles_only_in_one_counter(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 3

        result = visit_counter1 + visit_counter2

        assert result == {
            (0, 0): 4,
            (1, 1): 2,
        }


class TestAvg:
    def test_avg(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 3
        visit_counter2[(1, 1)] = 4

        result = VisitCounter.avg(visit_counter1, visit_counter2)

        assert result == {
            (0, 0): 2.0,
            (1, 1): 3.0,
        }

    def test_handles_only_in_one_counter(self):
        visit_counter1 = VisitCounter()
        visit_counter1[(0, 0)] = 1
        visit_counter1[(1, 1)] = 2

        visit_counter2 = VisitCounter()
        visit_counter2[(0, 0)] = 3

        result = VisitCounter.avg(visit_counter1, visit_counter2)

        assert result == {
            (1, 1): 1.0,
            (0, 0): 2.0,
        }
