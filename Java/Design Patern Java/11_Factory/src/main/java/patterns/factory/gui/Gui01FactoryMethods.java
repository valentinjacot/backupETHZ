package patterns.factory.gui;

import patterns.factory.gui.Components.Button;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;
import patterns.factory.gui.Components.Label;

public class Gui01FactoryMethods {

	public static void main(String[] args) {
		if (args.length > 0) showCalculator(args[0]);
		else showCalculator("Swing");
	}

	private static void showCalculator(String version) {
		Frame f = newFrame(version, "Calculator");
		final Field x = newField(version, 10, true);
		final Field y = newField(version, 10, true);
		final Field sum = newField(version, 10, false);
		Button b = newButton(version, "Compute", 
			source -> {
				int ix = Integer.parseInt(x.getText());
				int iy = Integer.parseInt(y.getText());
				sum.setText("" + (ix + iy));
			}
		);
		f.setGrid(4, 2);
		f.add(newLabel(version, "x"));
		f.add(x);
		f.add(newLabel(version, "y"));
		f.add(y);
		f.add(newLabel(version, "Summe"));
		f.add(sum);
		f.add(b);
		f.setVisible(true);
	}

	static private Frame newFrame(String version, String title) {
		switch (version) {
		case "AWT":
			return new ComponentsAWT.FrameAWT(title);
		case "Swing":
			return new ComponentsSwing.FrameSwing(title);
		case "SWT":
			return new ComponentsSWT.FrameSWT(title);
		case "FX":
			return new ComponentsFX.FrameFX(title);
		default:
			throw new IllegalStateException();
		}
	}

	static private Field newField(String version, int width, boolean enabled) {
		switch (version) {
		case "AWT":
			return new ComponentsAWT.FieldAWT(width, enabled);
		case "Swing":
			return new ComponentsSwing.FieldSwing(width, enabled);
		case "SWT":
			return new ComponentsSWT.FieldSWT(enabled);
		case "FX":
			return new ComponentsFX.FieldFX(width, enabled);
		default:
			throw new IllegalStateException();
		}
	}

	static private Button newButton(String version, String label,
			Components.ActionListener listener) {
		switch (version) {
		case "AWT":
			return new ComponentsAWT.ButtonAWT(label, listener);
		case "Swing":
			return new ComponentsSwing.ButtonSwing(label, listener);
		case "SWT":
			return new ComponentsSWT.ButtonSWT(label, listener);
		case "FX":
			return new ComponentsFX.ButtonFX(label, listener);
		default:
			throw new IllegalStateException();
		}
	}

	static private Label newLabel(String version, String text) {
		switch (version) {
		case "AWT":
			return new ComponentsAWT.LabelAWT(text);
		case "Swing":
			return new ComponentsSwing.LabelSwing(text);
		case "SWT":
			return new ComponentsSWT.LabelSWT(text);
		case "FX":
			return new ComponentsFX.LabelFX(text);
		default:
			throw new IllegalStateException();
		}
	}
}
