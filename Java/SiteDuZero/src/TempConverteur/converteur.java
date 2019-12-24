package TempConverteur;

import java.util.Scanner;

public class converteur {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		char direction;
		char retry ;
		Scanner sc = new Scanner(System.in);
		double temp;
		double C, F;
		do {
			do {
				System.out.println("Choose conversion direction");
				System.out.println("1 - Celsius to Fahrenheit");
				System.out.println("2 - Fahrenheit to Celsius ");
				direction = sc.nextLine().charAt(0);
			}while(direction != '1' && direction != '2' );
			System.out.println("Temperature to convert: ");
			temp = sc.nextDouble();
			sc.nextLine();
			switch(direction){
			case '1':
				System.out.println(temp + " degree Celsius is " +  9*temp / 5 + 32 + " degree Fahrenheit");
				break;
			case '2':
				System.out.println(temp + " degree Fahrenheit is " + ((temp-32)*5)/9 + " degree Celsius");
				break;
			}
			do {
				System.out.println("Would you like to retry(y/n) ?");
				retry= sc.nextLine().charAt(0);
			}while(retry != 'y' && retry != 'n');
		}while(retry == 'y');
		System.out.println("Bye");
	}

}
