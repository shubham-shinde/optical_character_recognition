function setup() {
   createCanvas(700, 700);
   background(0);
}



function mouseDragged() {
   strokeWeight(30);
   stroke(255);
   line(mouseX, mouseY, pmouseX, pmouseY);
}

/*function keyPressed() {
  if (keyCode == 32) {
    save("artwork.svg");
  }
}*/
