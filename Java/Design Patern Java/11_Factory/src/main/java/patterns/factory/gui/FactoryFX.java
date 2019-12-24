package patterns.factory.gui;

import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Button;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;
import patterns.factory.gui.Components.Label;
import patterns.factory.gui.ComponentsFX.ButtonFX;
import patterns.factory.gui.ComponentsFX.FieldFX;
import patterns.factory.gui.ComponentsFX.FrameFX;
import patterns.factory.gui.ComponentsFX.LabelFX;

public class FactoryFX implements ComponentFactory {
	@Override
	public Button newButton(final String label, final ActionListener listener) {
		return new ButtonFX(label, listener);
	}

	@Override
	public Label newLabel(final String label) {
		return new LabelFX(label);
	}

	@Override
	public Field newField(int width, boolean enabled) {
		return new FieldFX(width, enabled);
	}

	@Override
	public Frame newFrame(final String title) {
		return new FrameFX(title);
	}
	static {
		CurrentFactory.setFactory(new FactoryFX());
	}
}
