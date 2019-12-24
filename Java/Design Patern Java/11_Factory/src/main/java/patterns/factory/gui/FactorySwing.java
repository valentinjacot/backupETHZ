package patterns.factory.gui;

public class FactorySwing implements ComponentFactory {
	@Override
	public Components.Button newButton(final String label, Components.ActionListener listener) {
		return new ComponentsSwing.ButtonSwing(label, listener);
	}

	@Override
	public Components.Label newLabel(final String label) {
		return new ComponentsSwing.LabelSwing(label);
	}

	@Override
	public Components.Field newField(int width, boolean enabled) {
		return new ComponentsSwing.FieldSwing(width, enabled);
	}

	@Override
	public Components.Frame newFrame(String title) {
		return new ComponentsSwing.FrameSwing(title);
	}

	static {
		CurrentFactory.setFactory(new FactorySwing());
	}
}
