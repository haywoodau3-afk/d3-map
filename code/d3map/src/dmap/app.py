from __future__ import annotations

import sys
from pathlib import Path

from .errors import DMapError
from .release import create_release_project, default_structure_setup


def main() -> int:
    """Start the optional cross-platform PySide6 desktop application."""
    try:
        from PySide6.QtCore import QProcess, QSettings, QTimer, QUrl, Signal
        from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print(
            "d3map desktop requires the GUI extra: pip install 'd3map-catalysis[gui]'",
            file=sys.stderr,
        )
        return 2

    from .io import read_xyz_ensemble

    class StructureCanvas(QWidget):
        atom_selected = Signal(int)

        def __init__(self) -> None:
            super().__init__()
            self.setMinimumSize(420, 360)
            self._atoms: list[tuple[str, float, float]] = []
            self._points: list[tuple[float, float]] = []

        def load(self, path: str) -> None:
            geometry = read_xyz_ensemble(path).geometries[0]
            coordinates = geometry.coordinates
            centre = coordinates.mean(axis=0)
            shifted = coordinates - centre
            scale = max(float(abs(shifted[:, :2]).max()), 1.0)
            self._atoms = [
                (element, float(x / scale), float(y / scale))
                for element, (x, y, _z) in zip(geometry.elements, shifted, strict=True)
            ]
            self.update()

        def paintEvent(self, _event: object) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#101827"))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._points = []
            radius = 9.0
            for index, (element, x, y) in enumerate(self._atoms, start=1):
                px = self.width() / 2 + x * self.width() * 0.38
                py = self.height() / 2 - y * self.height() * 0.38
                self._points.append((px, py))
                colour = "#ff9d2e" if element in {"Fe", "Co", "Ni", "Cu", "Zn"} else "#78b7ff"
                if element == "H":
                    colour, radius = "#8793a3", 5.0
                else:
                    radius = 9.0
                painter.setBrush(QColor(colour))
                painter.setPen(QPen(QColor("#dfe7f2")))
                painter.drawEllipse(
                    int(px - radius), int(py - radius), int(2 * radius), int(2 * radius)
                )
                painter.drawText(int(px + radius + 2), int(py + 4), str(index))

        def mousePressEvent(self, event: object) -> None:
            position = event.position()
            if not self._points:
                return
            index, point = min(
                enumerate(self._points, start=1),
                key=lambda item: (item[1][0] - position.x()) ** 2
                + (item[1][1] - position.y()) ** 2,
            )
            distance = (point[0] - position.x()) ** 2 + (point[1] - position.y()) ** 2
            if distance <= 24**2:
                self.atom_selected.emit(index)

    class StructureForm(QGroupBox):
        def __init__(self, title: str, settings: object) -> None:
            super().__init__(title)
            self.settings = settings
            form = QFormLayout(self)
            file_row = QHBoxLayout()
            self.path = QLineEdit()
            browse = QPushButton("Browse")
            browse.clicked.connect(self._browse_xyz)
            file_row.addWidget(self.path)
            file_row.addWidget(browse)
            form.addRow("XYZ structure", file_row)
            self.center = QSpinBox()
            self.center.setRange(1, 99999)
            self.charge = QSpinBox()
            self.charge.setRange(-20, 20)
            self.multiplicity = QSpinBox()
            self.multiplicity.setRange(1, 20)
            self.alignment_atoms = QLineEdit("1,2,3")
            self.reactive = QLineEdit("0,0,1")
            self.secondary = QLineEdit("1,0,0")
            self.protected = QLineEdit()
            self.frozen = QLineEdit()
            self.steric = QLineEdit()
            self.solvent = QLineEdit()
            self.confirmed = QCheckBox("I reviewed the chemical setup and catalytic frame")
            for label, widget in (
                ("Catalytic centre atom", self.center),
                ("Charge", self.charge),
                ("Multiplicity", self.multiplicity),
                ("Alignment atoms", self.alignment_atoms),
                ("Reactive direction", self.reactive),
                ("Secondary direction", self.secondary),
                ("Protected contacts", self.protected),
                ("Frozen atoms", self.frozen),
                ("Steric atoms", self.steric),
                ("Solvent", self.solvent),
            ):
                form.addRow(label, widget)
            form.addRow(self.confirmed)
            self.canvas = StructureCanvas()
            self.canvas.atom_selected.connect(self.center.setValue)
            form.addRow("Click an atom to set the centre", self.canvas)

        def _browse_xyz(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Select XYZ structure", "", "XYZ structures (*.xyz);;All files (*)"
            )
            if selected:
                self.path.setText(selected)
                try:
                    self.canvas.load(selected)
                except (DMapError, OSError, ValueError) as error:
                    QMessageBox.warning(self, "Invalid structure", str(error))

        @staticmethod
        def _integers(text: str) -> list[int]:
            return [int(value.strip()) for value in text.split(",") if value.strip()]

        @staticmethod
        def _vector(text: str) -> list[float]:
            values = [float(value.strip()) for value in text.split(",") if value.strip()]
            if len(values) != 3:
                raise DMapError("frame directions require exactly three numbers")
            return values

        def setup(self, xtb: str | None, crest: str | None, threads: int) -> dict[str, object]:
            contacts = []
            for item in self.protected.text().split(","):
                if item.strip():
                    contacts.append([int(value) for value in item.split("-", 1)])
            steric = self._integers(self.steric.text()) or None
            return default_structure_setup(
                self.path.text(),
                center_atom=self.center.value(),
                reactive_direction=self._vector(self.reactive.text()),
                secondary_direction=self._vector(self.secondary.text()),
                alignment_atoms=self._integers(self.alignment_atoms.text()),
                charge=self.charge.value(),
                multiplicity=self.multiplicity.value(),
                solvent=self.solvent.text().strip() or None,
                confirmed=self.confirmed.isChecked(),
                protected_contacts=contacts,
                frozen_atoms=self._integers(self.frozen.text()),
                steric_atoms=steric,
                xtb_executable=xtb,
                crest_executable=crest,
                threads=threads,
            )

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("d3map")
            self.resize(1080, 820)
            self.settings = QSettings("d3map", "d3map")
            self.process: object | None = None
            self.project: Path | None = None
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            controls = QGridLayout()
            self.mode = QComboBox()
            self.mode.addItems(["d2-map", "d3-map"])
            self.mode.currentTextChanged.connect(self._mode_changed)
            self.xtb = QLineEdit(str(self.settings.value("xtb", "xtb")))
            self.crest = QLineEdit(str(self.settings.value("crest", "crest")))
            self.threads = QSpinBox()
            self.threads.setRange(1, 256)
            self.threads.setValue(1)
            controls.addWidget(QLabel("Analysis mode"), 0, 0)
            controls.addWidget(self.mode, 0, 1)
            controls.addWidget(QLabel("xTB executable"), 1, 0)
            controls.addWidget(self.xtb, 1, 1)
            controls.addWidget(QLabel("CREST executable"), 1, 2)
            controls.addWidget(self.crest, 1, 3)
            controls.addWidget(QLabel("Threads"), 0, 2)
            controls.addWidget(self.threads, 0, 3)
            layout.addLayout(controls)
            self.tabs = QTabWidget()
            self.reference = StructureForm("Reference structure", self.settings)
            self.extended = StructureForm("Extended structure", self.settings)
            self.tabs.addTab(self.reference, "Reference")
            self.tabs.addTab(self.extended, "Extended")
            layout.addWidget(self.tabs)
            buttons = QHBoxLayout()
            self.create_button = QPushButton("Create project")
            self.run_button = QPushButton("Run")
            self.cancel_button = QPushButton("Cancel")
            self.open_button = QPushButton("Open report")
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.open_button.setEnabled(False)
            self.create_button.clicked.connect(self._create)
            self.run_button.clicked.connect(self._run)
            self.cancel_button.clicked.connect(self._cancel)
            self.open_button.clicked.connect(self._open_report)
            for button in (self.create_button, self.run_button, self.cancel_button, self.open_button):
                buttons.addWidget(button)
            layout.addLayout(buttons)
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log)
            self._mode_changed(self.mode.currentText())

        def _mode_changed(self, mode: str) -> None:
            self.tabs.setTabEnabled(1, mode == "d3-map")

        def _create(self) -> None:
            try:
                output = QFileDialog.getExistingDirectory(self, "Choose parent output folder")
                if not output:
                    return
                reference_path = self.reference.path.text().strip()
                extended_path = self.extended.path.text().strip() if self.mode.currentText() == "d3-map" else None
                if not reference_path or (self.mode.currentText() == "d3-map" and not extended_path):
                    raise DMapError("select every required XYZ structure")
                project_name = (
                    f"{Path(reference_path).stem}-d2map"
                    if extended_path is None
                    else f"{Path(reference_path).stem}-vs-{Path(extended_path).stem}-d3map"
                )
                self.project = create_release_project(
                    kind=self.mode.currentText(),
                    reference_xyz=reference_path,
                    extended_xyz=extended_path,
                    output_directory=Path(output) / project_name,
                    reference_setup=self.reference.setup(
                        self.xtb.text().strip() or None,
                        self.crest.text().strip() or None,
                        self.threads.value(),
                    ),
                    extended_setup=(
                        None
                        if extended_path is None
                        else self.extended.setup(
                            self.xtb.text().strip() or None,
                            self.crest.text().strip() or None,
                            self.threads.value(),
                        )
                    ),
                )
                self.settings.setValue("xtb", self.xtb.text())
                self.settings.setValue("crest", self.crest.text())
                self.log.append(f"Created {self.project}")
                self.run_button.setEnabled(True)
            except (DMapError, OSError, ValueError) as error:
                QMessageBox.critical(self, "Cannot create project", str(error))

        def _run(self) -> None:
            if self.project is None:
                return
            self.process = QProcess(self)
            self.process.setProgram(sys.executable)
            self.process.setArguments(["-m", "d3map", "run", str(self.project)])
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._read_process)
            self.process.finished.connect(self._finished)
            self.process.start()
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.log.append("Started d3map run")

        def _read_process(self) -> None:
            if self.process is not None:
                data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
                self.log.append(data.rstrip())

        def _finished(self, exit_code: int, _status: object) -> None:
            self.cancel_button.setEnabled(False)
            self.run_button.setEnabled(True)
            self.open_button.setEnabled(exit_code == 0)
            self.log.append(f"Run finished with exit code {exit_code}")

        def _cancel(self) -> None:
            if self.process is not None:
                self.process.terminate()
                if not self.process.waitForFinished(3000):
                    self.process.kill()
                self.log.append("Run cancelled; completed stages remain available for resumption")

        def _open_report(self) -> None:
            if self.project is None:
                return
            comparison = self.project.parent / "results" / "comparison" / "report.html"
            reference = self.project.parent / "results" / "reference" / "report.html"
            target = comparison if comparison.is_file() else reference
            if target.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

        def closeEvent(self, event: object) -> None:
            if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
                answer = QMessageBox.question(
                    self,
                    "Calculation active",
                    "Cancel the active calculation and close d3map?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self._cancel()
            event.accept()

    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("d3map")
    application.setOrganizationName("d3map")
    window = Window()
    window.show()
    if "--screenshot" in sys.argv:
        try:
            screenshot_path = Path(sys.argv[sys.argv.index("--screenshot") + 1])
        except IndexError:
            print("--screenshot requires an output path", file=sys.stderr)
            return 2

        def save_screenshot() -> None:
            window.grab().save(str(screenshot_path))
            application.quit()

        QTimer.singleShot(750, save_screenshot)
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(750, application.quit)
    return application.exec()
