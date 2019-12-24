package patterns.factory.gui;

public interface Components {
	/** Base type for all component interfaces. */
	interface Component {}

	/** A button with a label and an action listener. */
	interface Button extends Component {}

	/** A label with a text. */
	interface Label extends Component {}

	/** A text field where the text can be read and set. */
	interface Field extends Component {
		String getText();
		void setText(String text);
	}

	/** A program frame which controls components in a grid layout. */
	interface Frame {
		void setVisible(boolean visible);
		void add(Component c);
		void setGrid(int w, int h);
	}

	/** The listener interface for receiving action events. */
	interface ActionListener {
		void actionPerformed(Component source);
	}
}
