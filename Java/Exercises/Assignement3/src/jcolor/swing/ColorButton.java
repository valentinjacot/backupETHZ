package jcolor.swing;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import jcolor.ColorListener;
import jcolor.ColorModel;
public class ColorButton extends JButton implements ColorListener, ActionListener{
	enum Type { BRIGHTER, DARKER}
	private ColorModel model;
	private Type type;
	ColorButton(ColorModel model, Type type, String label){
		super(label);
		this.type=type;
		this.model=model;
		addActionListener(this);
		model.addColorListener(this);
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		Color c= model.getColor();
		switch(type) {
		case BRIGHTER: model.setColor(c.brighter());break;
		case DARKER: model.setColor(c.darker());break;
		}
	}
	@Override
	public void colorValueChanged(Color c) {
		switch(type){
		case BRIGHTER:setEnabled(!c.equals(c.brighter()));break;
		case DARKER:setEnabled(!c.equals(c.darker()));break;
		}
	}
}