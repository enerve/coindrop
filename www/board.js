

function draw_slot(cx, cy, id) {
	var svg = document.getElementsByTagName('svg')[0]; //Get svg element
	var slot = document.createElementNS("http://www.w3.org/2000/svg", 'circle');
	slot.setAttribute("cx", cx);
	slot.setAttribute("cy", cy);
	slot.setAttribute("r", 40);
	slot.style.fill = "#FFF"
	slot.style.stroke = "#000";
	slot.style.strokeWidth = "3px";
	slot.setAttribute("id", id);
	slot.setAttribute("onclick", "clicked(\"" + id +"\")");

	svg.appendChild(slot);}

function draw_grid() {
	var svg = document.getElementsByTagName('svg')[0]; //Get svg element

	rect_x = 300
	rect_y = 100
	rect_width = 650
	rect_height = 550
	border = 10
	
	var rect = document.createElementNS("http://www.w3.org/2000/svg", 'rect');
	rect.setAttribute("x", rect_x);
	rect.setAttribute("y", rect_y);
	rect.setAttribute("rx", 10);
	rect.setAttribute("ry", 10);
	rect.setAttribute("width", rect_width);
	rect.setAttribute("height", rect_height);
	rect.style.fill = "#00F"
		rect.style.stroke = "#000";
	rect.style.strokeWidth = "3px";
	svg.appendChild(rect);

	rw = rect_width - 2 * border
	rh = rect_height - 2 * border
	
	swidth = rw / 7
	sheight = rh / 6
	
	for (i=0; i<7; i++) {
		for (j=0; j<6; j++) {
			draw_slot(rect_x + border + i * swidth + swidth/2,
					rect_y + border + j * sheight + sheight/2,
					"slot" + i + "" + j)
		}
	}
}

function draw_board(B) {
	draw_grid()
}

Game = function() {
	this.B = new Array(6);
	for (var i = 0; i < this.B.length; i++) {
	  this.B[i] = new Array(7).fill(0);
	}

	this.turn = 1
};
	
Game.prototype.clicked = function(x, y) {
	if (this.B[y][x]==0) {
		// okay
		
		this.B[y][x] = this.turn
		placeCoin('slot'+x+y, this.turn)
		
		this.turn = -this.turn;
		
		return true;
	} else {
		// can't
		alert('not ok - ' + this.B[y][x])
		return false;
	}
};

function placeCoin(id, player) {
	var slot = document.getElementById(id); //Get svg element
	
	if (player == 1) {
		color="#F00" 
	} else {
		color="#ED0"
	}
		
	slot.style.fill=color
	
}

game = new Game()

function clicked(id) {
	x = parseInt(id[4])
	y = parseInt(id[5])
	
	if (game.clicked(x, y)) {
		//placeCoin(id, 1)
	}
	
}

