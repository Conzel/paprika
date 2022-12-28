from PyQt5.QtGui import QFontDatabase, QGuiApplication
import sys

gui_app = QGuiApplication(sys.argv)
database = QFontDatabase()
print(database.families(QFontDatabase.Latin))