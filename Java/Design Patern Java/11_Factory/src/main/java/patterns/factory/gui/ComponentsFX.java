package patterns.factory.gui;

import java.util.LinkedList;
import java.util.List;

import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Component;

public class ComponentsFX {
	private static boolean initialized = false;

	private static void initializeToolkit() {
		if (!initialized) {
			initialized = true;
			new JFXPanel(); // this will prepare JavaFX toolkit and environment
			// Reference:
			// http://stackoverflow.com/questions/11273773/javafx-2-1-toolkit-not-initialized}
		}
	}

	public static class ButtonFX extends javafx.scene.control.Button implements Components.Button {
		ButtonFX(String label, ActionListener listener) {
			super(label);
			initializeToolkit();
			setOnAction(new EventHandler<ActionEvent>() {
				@Override
				public void handle(ActionEvent event) {
					listener.actionPerformed(ButtonFX.this);
				}
			});
		}
	}

	public static class LabelFX extends javafx.scene.control.Label implements Components.Label {
		LabelFX(String label) {
			super(label);
			initializeToolkit();
		}
	}

	static class FieldFX extends TextField implements Components.Field {
		public FieldFX(int width, boolean enabled) {
			initializeToolkit();
			setPrefColumnCount(width);
			setEditable(enabled);
			setFocusTraversable(enabled);
		}
	}

	public static class FrameFX implements Components.Frame {
		private String title;
		private List<Component> components = new LinkedList<>();
		private int w, h;

		public FrameFX(String title) {
			initializeToolkit();
			this.title = title;
		}

		public void init() {
			System.out.println("init called");
		}

		@Override
		public void add(Component c) {
			components.add(c);
		}

		@Override
		public void setGrid(int h, int w) {
			this.w = w;
			this.h = h;
		}

		public @Override void setVisible(boolean visible) {
			Platform.runLater(() -> {
				final Stage s = new Stage();
				s.setTitle(title);
				s.setResizable(false);
				Group root = new javafx.scene.Group();
				Scene scene = new Scene(root);
				GridPane gridpane = new GridPane();
				gridpane.setPadding(new Insets(5));
				gridpane.setHgap(5);
				gridpane.setVgap(5);
				int i = 0;
				for (Component c : components) {
					gridpane.add((Node) c, i % w, i / w);
					i++;
					if (i == w * h)
						break;
				}
				root.getChildren().add(gridpane);
				s.setScene(scene);
				s.setOnCloseRequest(new EventHandler<WindowEvent>() {
					@Override
					public void handle(WindowEvent event) {
						Platform.runLater(() -> Platform.exit());
					}
				});
				s.show();
			});
		}
	}
}
