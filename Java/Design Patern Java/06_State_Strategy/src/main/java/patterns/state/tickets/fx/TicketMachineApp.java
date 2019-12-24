package patterns.state.tickets.fx;

import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class TicketMachineApp {

	public void start(Stage primaryStage) {
		TicketMachine m = new TicketMachine1();
		Scene scene = new Scene(new TicketMachineGui(m));
		primaryStage.setTitle("VBZ Ticketmachine");
		primaryStage.setScene(scene);
		primaryStage.show();
		primaryStage.setHeight(4*174);
		primaryStage.setWidth(4*141);
	}

	public static void main(String[] args) {
		new JFXPanel();
		Platform.runLater(() -> {
			TicketMachineApp app = new TicketMachineApp();
			app.start(new Stage());
		});
	}
}
