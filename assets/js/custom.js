let height_all = $('#mvsvbrdf').outerHeight(true);
let height_title = $('#mvsvbrdf_title').outerHeight(true); // 64
let height_author = $('#mvsvbrdf_author').outerHeight(true); // 64
let height_links = $('#mvsvbrdf_links').outerHeight(false); // 38
let height_desire = ( height_all - height_title - height_author - height_links );
let options = {
	height: height_desire,
	keep: '.toggle',
};
let wrappers = document.querySelectorAll( ".element-to-truncate" );

document.addEventListener( "DOMContentLoaded", () => {	
	for(var i=0;i< wrappers.length;++i){
		let wrapper = wrappers[i];
		let dot = new Dotdotdot( wrapper, options );
		let api = dot.API;
		wrapper.addEventListener( 'click', ( evnt ) => {
			if ( evnt.target.closest( '.toggle' ) ) {
				evnt.preventDefault();
				//	When truncated, restore
				if ( wrapper.matches( '.ddd-truncated' ) )
				{
					api.restore();
					wrapper.classList.add( 'full-story' );
				}
				//	Not truncated, truncate
				else
				{
					wrapper.classList.remove( 'full-story' );
					api.truncate();
					api.watch();
				}
			}
		});
	}
});


function lastModified() {
    var modiDate = new Date(document.lastModified);
    var showAs = modiDate.getDate() + "-" + (modiDate.getMonth() + 1) + "-" + modiDate.getFullYear();
    return showAs
}

$(document).ready(function () {$("#lastModified").text('Last modified on ' + document.lastModified);});

// // for nav jump
// var height_h3 = $('#title_h3').outerHeight(true) * 0.9;
// var shiftWindow = function() { scrollBy(0, -height_h3); };
// if (window.location.hash) shiftWindow();
// window.addEventListener("hashchange", shiftWindow);
