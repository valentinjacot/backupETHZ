package patterns.state.tickets.fx;

import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundImage;
import javafx.scene.layout.BackgroundPosition;
import javafx.scene.layout.BackgroundRepeat;
import javafx.scene.layout.BackgroundSize;
import javafx.scene.layout.Pane;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

public class TicketMachineGui extends Pane {
	private final TicketMachine m; 
	
	private SimpleStringProperty label = new SimpleStringProperty("");
	private SimpleStringProperty price = new SimpleStringProperty("");

	private SimpleBooleanProperty initState = new SimpleBooleanProperty();
	private SimpleBooleanProperty destSelectedState = new SimpleBooleanProperty();
	private SimpleBooleanProperty moneyEnteredState = new SimpleBooleanProperty();
	
	private void update() {
		if(m.isInStateInit() && price.get().equals("") && label.get().length() >= 4) {
			m.setDestination(Integer.parseInt(label.get()));
			price.set(String.format("%2.2f", m.getPrice()));
		} else if (m.isInStateInit() && !price.get().equals("") ) {
			price.set("");
			label.set("");
		} else if (m.isInStateInit()) {
			price.set("");
		} else {
			price.set(String.format("%2.2f", m.getPrice()-m.getEnteredMoney()));
		}
		initState.set(m.isInStateInit());
		destSelectedState.set(m.isInStateDestSelected());
		moneyEnteredState.set(m.isInStateMoneyEntered());
	}

	public TicketMachineGui(TicketMachine m) {
		this.m = m;
		Image image = new Image("Ticketmachine2.png");
		BackgroundSize backgroundSize = new BackgroundSize(100, 100, true, true, false, true);
		BackgroundImage backgroundImage = new BackgroundImage(image, BackgroundRepeat.NO_REPEAT, BackgroundRepeat.NO_REPEAT, BackgroundPosition.CENTER, backgroundSize);
		Background background = new Background(backgroundImage);
		this.setBackground(background);
		this.setWidth(2*141);
		this.setHeight(2*174);
		
		ImageView money = new ImageView(new Image("Money.png"));
		money.visibleProperty().bind(initState.not());
		money.setLayoutX(253);
		money.setLayoutY(333);
		money.setOnMouseClicked(e -> {
			double x = e.getX();
			double y = e.getY();
			if((x-60)*(x-60) + (y-60)*(y-60) < 50*50) {
				m.enterMoney(5);
			} else if((x-148)*(x-148) + (y-60)*(y-60) < 43*43) {
				m.enterMoney(2);
			} else if((x-188)*(x-188) + (y-124)*(y-124) < 38*38) {
				m.enterMoney(1);
			} else if((x-185)*(x-185) + (y-184)*(y-184) < 30*30) {
				m.enterMoney(0.5);
			} else if((x-140)*(x-140) + (y-216)*(y-216) < 34*34) {
				m.enterMoney(0.2);
			} else if((x-87)*(x-87) + (y-234)*(y-234) < 29*29) {
				m.enterMoney(0.1);
			} else if((x-36)*(x-36) + (y-219)*(y-219) < 27*27) {
				m.enterMoney(0.05);
			}
			update();
		});
		
		TextField destField = new TextField("");
		destField.setLayoutX(89);
		destField.setLayoutY(170);
		destField.setPrefWidth(242-89);
		destField.setPrefHeight(40);
		destField.textProperty().bind(label);
		destField.setOnInputMethodTextChanged(e -> update());
		destField.setEditable(false);
		destField.textProperty().addListener(e -> update());
		destField.setFont(Font.font("Arial", FontWeight.BOLD, 20));
		destField.setAlignment(Pos.BASELINE_RIGHT);
		
		TextField moneyField = new TextField("");
		moneyField.setLayoutX(90);
		moneyField.setLayoutY(88);
		moneyField.setPrefWidth(242-89);
		moneyField.setPrefHeight(50);
		moneyField.textProperty().bind(price);
		moneyField.setEditable(false);
		moneyField.visibleProperty().bind(initState.not());
		moneyField.setFont(Font.font("Consolas", FontWeight.BOLD, 30));
		moneyField.setAlignment(Pos.BASELINE_RIGHT);
		
		Button b1 = newButton("1", 116, 461);
		Button b2 = newButton("2", 154, 461);
		Button b3 = newButton("3", 192, 461);
		Button b4 = newButton("4", 116, 503);
		Button b5 = newButton("5", 154, 503);
		Button b6 = newButton("6", 192, 503);
		Button b7 = newButton("7", 116, 545);
		Button b8 = newButton("8", 154, 545);
		Button b9 = newButton("9", 192, 545);

	
		Button first = new Button("1. Kl");
		first.setLayoutX(88);
		first.setLayoutY(253);
		first.setPrefWidth(45);
		first.setPrefHeight(30);
		first.setOnAction(e -> { m.setFirstClass(!m.isFirstClass()); update(); });
		first.disableProperty().bind(destSelectedState.not());
		
		Button retour = new Button("RET");
		retour.setLayoutX(144);
		retour.setLayoutY(253);
		retour.setPrefWidth(45);
		retour.setPrefHeight(30);
		retour.setOnAction(e -> { m.setDayTicket(!m.isDayTicket()); update(); });
		retour.disableProperty().bind(destSelectedState.not());
		
		Button half = new Button("1/2");
		half.setLayoutX(201);
		half.setLayoutY(253);
		half.setPrefWidth(45);
		half.setPrefHeight(30);
		half.setOnAction(e -> { m.setHalfPrice(!m.isHalfPrice()); update(); }); 
		half.disableProperty().bind(destSelectedState.not());
		
		this.getChildren().addAll(b1,b2,b3,b4,b5,b6,b7,b8,b9, first, retour, half, moneyField, destField, money);
		
		update();
		
//		this.setOnMouseClicked(e -> System.out.println(e));
	}

	private Button newButton(String value, int x, int y) {
		Button button = new Button(value);
		button.setLayoutX(x);
		button.setLayoutY(y);
		button.setPrefWidth(28);
		button.setPrefHeight(28);
		button.setOnAction(e -> label.set(label.get() + value));
		button.disableProperty().bind(initState.not());
		return button;
	}

}
