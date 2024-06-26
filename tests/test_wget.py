import unittest
from subprocess import check_output


class Testwget(unittest.TestCase):
    def test_wget(self):
        if "no wget" in check_output(["which", "wget"]).decode():
            raise EnvironmentError("wget not found")

if __name__ == '__main__':
    unittest.main()
