from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget


class Toast(QLabel):
    _STYLES = {
        "info":    "background-color: #0078D4; color: white;",
        "success": "background-color: #107C10; color: white;",
        "warning": "background-color: #C77700; color: white;",
        "error":   "background-color: #C42B1C; color: white;",
    }

    def __init__(self, parent: QWidget, message: str, level: str = "info", duration_ms: int = 4000):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setMaximumWidth(420)
        self.setText(message)
        self.setStyleSheet(
            "QLabel {"
            + self._STYLES.get(level, self._STYLES["info"])
            + "border-radius: 8px; padding: 10px 16px; font-weight: bold;"
            "}"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.adjustSize()
        self.show()
        self.raise_()

        QTimer.singleShot(duration_ms, self._fade_out)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()


class ToastManager:
    MARGIN = 16
    SPACING = 8

    def __init__(self, parent: QWidget):
        self._parent = parent
        self._toasts: list[Toast] = []

    def show(self, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        toast = Toast(self._parent, message, level, duration_ms)
        self._toasts.append(toast)
        toast.destroyed.connect(lambda *_: self._forget(toast))
        self.reposition()

    def _forget(self, toast: Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)

    def reposition(self) -> None:
        y = self.MARGIN
        parent_width = self._parent.width()
        for toast in self._toasts:
            toast.adjustSize()
            x = max(self.MARGIN, parent_width - toast.width() - self.MARGIN)
            toast.move(x, y)
            toast.raise_()
            y += toast.height() + self.SPACING

class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._full_text = text
        self.setToolTip(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._apply_elide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        metrics = self.fontMetrics()
        elided = metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, max(0, self.width())
        )
        super().setText(elided)