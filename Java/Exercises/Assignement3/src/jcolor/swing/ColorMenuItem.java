package jcolor.swing;
import java.awt.Color;
import javax.swing.JCheckBoxMenuItem;
import jcolor.ColorModel;
public class ColorMenuItem extends JCheckBoxMenuItem {
	ColorMenuItem(ColorModel model, String label, Color color){
		super (label);
		addActionListener(e -> model.setColor(color));
		model.addColorListener(c->setSelected(c.equals(color)));
	}
}