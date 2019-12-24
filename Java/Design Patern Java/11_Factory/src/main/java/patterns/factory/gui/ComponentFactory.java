package patterns.factory.gui;

public interface ComponentFactory {
	/** Creates a button with a given label and listener. */
	Components.Button newButton(String label, Components.ActionListener listener);

	/** Creates a label with a given name. */
	Components.Label newLabel(String label);

	/** Creates a text field. The returned result implements interface Field. */
	Components.Field newField(int width, boolean enabled);

	/** Creates a frame with a given title. */
	Components.Frame newFrame(String title);
}
