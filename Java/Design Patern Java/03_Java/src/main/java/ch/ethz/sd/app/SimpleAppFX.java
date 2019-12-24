package ch.ethz.sd.app;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;

public class SimpleAppFX extends Application {

	public static void main(String[] args) {
		launch(args);
	}

	@Override
	public void start(Stage primaryStage) {
		TextField txt = new TextField();

		Button btn = new Button("Submit");
		btn.setOnAction(e -> { System.out.println("[FX] Sbumit: " + txt.getText()); });

		Pane center = new FlowPane(5,0);
		center.getChildren().add(txt);
		center.getChildren().add(btn);

		MenuItem fileMenuItem = new MenuItem("File ...");
		MenuItem exitMenuItem = new MenuItem("Exit");
		exitMenuItem.setOnAction(e -> Platform.exit());
		Menu menu = new Menu("File", null, fileMenuItem, new SeparatorMenuItem(), exitMenuItem);

		MenuBar menuBar = new MenuBar(menu);
		BorderPane root = new BorderPane(center);
		root.setTop(menuBar);

		Scene scene = new Scene(root, 220, 50);

		primaryStage.setTitle("Simple Application");
		primaryStage.setScene(scene);
		primaryStage.show();
	}

}
