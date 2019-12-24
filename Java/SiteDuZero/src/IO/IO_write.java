package IO;

//Packages � importer afin d'utiliser l'objet File
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class IO_write {
public static void main(String[] args) {
  //Nous d�clarons nos objets en dehors du bloc try/catch
  FileInputStream fis;
  FileOutputStream fos;
  BufferedInputStream bis;
  BufferedOutputStream bos; 
      
  try {
    fis = new FileInputStream(new File("dictionnaire.txt"));
    fos = new FileOutputStream(new File("test.txt"));
    bis = new BufferedInputStream(new FileInputStream(new File("dictionnaire2.txt")));
    bos = new BufferedOutputStream(new FileOutputStream(new File("test3.txt")));
    byte[] buf = new byte[8];

    //On r�cup�re le temps du syst�me
    long startTime = System.currentTimeMillis();
              
    while(fis.read(buf) != -1){
      fos.write(buf);
    }
    //On affiche le temps d'ex�cution
    System.out.println("Temps de lecture + �criture avec FileInputStream et FileOutputStream : " + (System.currentTimeMillis() - startTime));
              
    //On r�initialise                
    startTime = System.currentTimeMillis();

    while(bis.read(buf) != -1){
      bos.write(buf);
    }
    //On r�affiche
    System.out.println("Temps de lecture + �criture avec BufferedInputStream et BufferedOutputStream : " + (System.currentTimeMillis() - startTime));
              
    //On ferme nos flux de donn�es
    fis.close();
    bis.close();
    fos.close();
    bos.close();
              
  } catch (FileNotFoundException e) {
    e.printStackTrace();
  } catch (IOException e) {
    e.printStackTrace();
  }     	
}
}