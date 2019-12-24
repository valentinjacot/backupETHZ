package patterns.factory.gui;

public class FactoryAWT implements ComponentFactory {
	@Override
	public Components.Button newButton(String label, Components.ActionListener listener) {
		return new ComponentsAWT.ButtonAWT(label, listener);
	}

	@Override
	public Components.Label newLabel(String label) {
		return new ComponentsAWT.LabelAWT(label);
	}

	@Override
	public Components.Field newField(int width, boolean enabled) {
		return new ComponentsAWT.FieldAWT(width, enabled);
	}

	@Override
	public Components.Frame newFrame(String title) {
		return new ComponentsAWT.FrameAWT(title);
	}

	static {
		CurrentFactory.setFactory(new FactoryAWT());
	}
}
