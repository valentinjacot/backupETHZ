package IO;

//Package � importer afin d'utiliser l'objet File
import java.io.File;

public class Main {
public static void main(String[] args) {
  //Cr�ation de l'objet File
  File f = new File("test.txt");
  System.out.println("Chemin absolu du fichier : " + f.getAbsolutePath());
  System.out.println("Nom du fichier : " + f.getName());
  System.out.println("Est-ce qu'il existe ? " + f.exists());
  System.out.println("Est-ce un r�pertoire ? " + f.isDirectory());
  System.out.println("Est-ce un fichier ? " + f.isFile());
   
  System.out.println("Affichage des lecteurs � la racine du PC : ");
  for(File file : f.listRoots())
  {
    System.out.println(file.getAbsolutePath());
    try {
      int i = 1; 
      //On parcourt la liste des fichiers et r�pertoires
      for(File nom : file.listFiles()){
         //S'il s'agit d'un dossier, on ajoute un "/"
         System.out.print("\t\t" +
                ((nom.isDirectory()) ? nom.getName()+"/" : nom.getName()));
                   
         if((i%4) == 0){
          System.out.print("\n");
         }
         i++;
      }
      System.out.println("\n");
    } catch (NullPointerException e) {
      //L'instruction peut g�n�rer une NullPointerException
      //s'il n'y a pas de sous-fichier !
    }
  }      
}
}