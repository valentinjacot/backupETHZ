package jcolor.swing;
import java.awt.Color;
import javax.swing.JRadioButton;
import jcolor.ColorModel;
class ColorRadioButton extends JRadioButton {
	ColorRadioButton(ColorModel model, String label, Color color) {
		super(label, false);
		addActionListener(e -> model.setColor(color));
		model.addColorListener(c -> setSelected(c.equals(color)));
	}
}